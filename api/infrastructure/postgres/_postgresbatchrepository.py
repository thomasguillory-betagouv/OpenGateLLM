from datetime import datetime, timedelta

from sqlalchemy import func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from api.domain.batch import BatchRepository
from api.domain.batch.entities import (
    Batch,
    BatchPage,
    BatchProcessingStatus,
    BatchRequest,
    BatchRequestCounts,
    BatchRequestError,
    BatchRequestStatus,
    BatchRequestToCreate,
    ClaimedBatchRequest,
)
from api.domain.batch.errors import BatchNotFoundError
from api.infrastructure.postgres._pagination import fetch_page_with_total
from api.infrastructure.postgres.models import Batch as BatchTable
from api.infrastructure.postgres.models import BatchRequest as BatchRequestTable

UNFINISHED_REQUEST_STATUSES = (BatchRequestStatus.PENDING, BatchRequestStatus.PROCESSING)


class PostgresBatchRepository(BatchRepository):
    """Every worker-side method is a single statement, so it stays atomic on an AutocommitSession: the worker never
    holds a transaction, nor a pooled connection, while a provider computes a completion."""

    def __init__(self, postgres_session: AsyncSession):
        self.postgres_session = postgres_session

    async def create_batch(self, user_id: int, key_id: int, expires: datetime, requests: list[BatchRequestToCreate]) -> Batch:
        result = await self.postgres_session.execute(
            insert(BatchTable)
            .values(user_id=user_id, key_id=key_id, processing_status=BatchProcessingStatus.IN_PROGRESS, expires=expires)
            .returning(BatchTable)
        )
        row = result.scalar_one()

        await self.postgres_session.execute(
            insert(BatchRequestTable),
            [
                {
                    "batch_id": row.id,
                    "custom_id": request.custom_id,
                    "position": position,
                    "params": request.params,
                    "status": BatchRequestStatus.PENDING,
                }
                for position, request in enumerate(requests)
            ],
        )

        return self._row_to_batch(row=row, request_counts=BatchRequestCounts(processing=len(requests)))

    async def get_batch(self, batch_id: int, user_id: int) -> Batch | BatchNotFoundError:
        row = await self.postgres_session.scalar(
            # the worker statements bypass the identity map: a batch loaded earlier in the session may be stale
            select(BatchTable).where(BatchTable.id == batch_id, BatchTable.user_id == user_id).execution_options(populate_existing=True)
        )
        if row is None:
            return BatchNotFoundError(id=batch_id)

        request_counts = await self._get_request_counts(batch_ids=[batch_id])
        return self._row_to_batch(row=row, request_counts=request_counts[batch_id])

    async def get_batches_page(self, user_id: int, offset: int, limit: int) -> BatchPage:
        filters = [BatchTable.user_id == user_id]
        page_query = select(BatchTable, func.count().over().label("total")).where(*filters).order_by(BatchTable.id.desc()).offset(offset).limit(limit)
        count_query = select(func.count()).select_from(BatchTable).where(*filters)
        rows, total = await fetch_page_with_total(self.postgres_session, page_query, count_query)

        request_counts = await self._get_request_counts(batch_ids=[row[0].id for row in rows])
        return BatchPage(total=total, data=[self._row_to_batch(row=row[0], request_counts=request_counts[row[0].id]) for row in rows])

    async def cancel_batch(self, batch_id: int, user_id: int) -> Batch | BatchNotFoundError:
        result = await self.postgres_session.execute(
            update(BatchTable)
            .where(BatchTable.id == batch_id, BatchTable.user_id == user_id, BatchTable.processing_status == BatchProcessingStatus.IN_PROGRESS)
            .values(processing_status=BatchProcessingStatus.CANCELING, cancel_initiated=func.now())
            .returning(BatchTable.id)
        )
        if result.scalar_one_or_none() is not None:
            await self.postgres_session.execute(
                update(BatchRequestTable)
                .where(BatchRequestTable.batch_id == batch_id, BatchRequestTable.status == BatchRequestStatus.PENDING)
                .values(status=BatchRequestStatus.CANCELED, completed=func.now())
            )

        return await self.get_batch(batch_id=batch_id, user_id=user_id)

    async def get_batch_requests(self, batch_id: int, after_position: int, limit: int) -> list[BatchRequest]:
        rows = await self.postgres_session.scalars(
            select(BatchRequestTable)
            .where(BatchRequestTable.batch_id == batch_id, BatchRequestTable.position > after_position)
            .order_by(BatchRequestTable.position)
            .limit(limit)
        )
        return [self._row_to_batch_request(row) for row in rows]

    async def claim_requests(self, limit: int, lease_seconds: int) -> list[ClaimedBatchRequest]:
        claimable = (
            select(BatchRequestTable.id)
            .join(BatchTable, BatchTable.id == BatchRequestTable.batch_id)
            .where(
                BatchRequestTable.status == BatchRequestStatus.PENDING,
                BatchRequestTable.available_at <= func.now(),
                BatchTable.expires > func.now(),
            )
            .order_by(BatchRequestTable.position, BatchRequestTable.batch_id)
            .limit(limit)
            .with_for_update(skip_locked=True, of=BatchRequestTable)
        )
        result = await self.postgres_session.execute(
            update(BatchRequestTable)
            .where(BatchRequestTable.id.in_(claimable), BatchRequestTable.batch_id == BatchTable.id)
            .values(status=BatchRequestStatus.PROCESSING, locked_until=func.now() + timedelta(seconds=lease_seconds))
            .returning(
                BatchRequestTable.id,
                BatchRequestTable.batch_id,
                BatchRequestTable.custom_id,
                BatchRequestTable.position,
                BatchRequestTable.params,
                BatchRequestTable.attempts,
                BatchTable.user_id,
                BatchTable.key_id,
            )
            .execution_options(synchronize_session=False)
        )
        return [
            ClaimedBatchRequest(
                id=row.id,
                batch_id=row.batch_id,
                custom_id=row.custom_id,
                position=row.position,
                params=row.params,
                status=BatchRequestStatus.PROCESSING,
                attempts=row.attempts,
                user_id=row.user_id,
                key_id=row.key_id,
            )
            for row in result.all()
        ]

    async def complete_request(self, request_id: int, status: BatchRequestStatus, response: dict | None, error: BatchRequestError | None) -> None:
        await self.postgres_session.execute(
            update(BatchRequestTable)
            .where(BatchRequestTable.id == request_id, BatchRequestTable.status == BatchRequestStatus.PROCESSING)
            .values(
                status=status,
                response=response,
                error=error.model_dump() if error is not None else None,
                locked_until=None,
                completed=func.now(),
            )
            .execution_options(synchronize_session=False)
        )

    async def requeue_request(self, request_id: int, delay_seconds: float, count_attempt: bool) -> None:
        values = {
            "status": BatchRequestStatus.PENDING,
            "locked_until": None,
            "available_at": func.now() + timedelta(seconds=delay_seconds),
        }
        if count_attempt:
            values["attempts"] = BatchRequestTable.attempts + 1

        await self.postgres_session.execute(
            update(BatchRequestTable)
            .where(BatchRequestTable.id == request_id, BatchRequestTable.status == BatchRequestStatus.PROCESSING)
            .values(**values)
            .execution_options(synchronize_session=False)
        )

    async def release_expired_claims(self) -> int:
        result = await self.postgres_session.execute(
            update(BatchRequestTable)
            .where(BatchRequestTable.status == BatchRequestStatus.PROCESSING, BatchRequestTable.locked_until < func.now())
            .values(status=BatchRequestStatus.PENDING, locked_until=None, available_at=func.now())
            .execution_options(synchronize_session=False)
        )
        return result.rowcount

    async def expire_overdue_requests(self) -> int:
        overdue_batch_ids = select(BatchTable.id).where(BatchTable.expires <= func.now(), BatchTable.processing_status != BatchProcessingStatus.ENDED)
        result = await self.postgres_session.execute(
            update(BatchRequestTable)
            .where(BatchRequestTable.status == BatchRequestStatus.PENDING, BatchRequestTable.batch_id.in_(overdue_batch_ids))
            .values(status=BatchRequestStatus.EXPIRED, completed=func.now())
            .execution_options(synchronize_session=False)
        )
        return result.rowcount

    async def end_finished_batches(self) -> int:
        has_unfinished_requests = (
            select(BatchRequestTable.id)
            .where(BatchRequestTable.batch_id == BatchTable.id, BatchRequestTable.status.in_(UNFINISHED_REQUEST_STATUSES))
            .exists()
        )
        result = await self.postgres_session.execute(
            update(BatchTable)
            .where(
                BatchTable.processing_status.in_((BatchProcessingStatus.IN_PROGRESS, BatchProcessingStatus.CANCELING)),
                ~has_unfinished_requests,
            )
            .values(processing_status=BatchProcessingStatus.ENDED, ended=func.now())
            .execution_options(synchronize_session=False)
        )
        return result.rowcount

    async def _get_request_counts(self, batch_ids: list[int]) -> dict[int, BatchRequestCounts]:
        request_counts = {batch_id: BatchRequestCounts() for batch_id in batch_ids}
        if not batch_ids:
            return request_counts

        rows = await self.postgres_session.execute(
            select(BatchRequestTable.batch_id, BatchRequestTable.status, func.count())
            .where(BatchRequestTable.batch_id.in_(batch_ids))
            .group_by(BatchRequestTable.batch_id, BatchRequestTable.status)
        )
        for batch_id, status, count in rows.all():
            counts = request_counts[batch_id]
            match status:
                case BatchRequestStatus.PENDING | BatchRequestStatus.PROCESSING:
                    counts.processing += count
                case BatchRequestStatus.SUCCEEDED:
                    counts.succeeded = count
                case BatchRequestStatus.ERRORED:
                    counts.errored = count
                case BatchRequestStatus.CANCELED:
                    counts.canceled = count
                case BatchRequestStatus.EXPIRED:
                    counts.expired = count

        return request_counts

    @staticmethod
    def _row_to_batch(row: BatchTable, request_counts: BatchRequestCounts) -> Batch:
        return Batch(
            id=row.id,
            user_id=row.user_id,
            key_id=row.key_id,
            processing_status=row.processing_status,
            request_counts=request_counts,
            created=row.created,
            expires=row.expires,
            ended=row.ended,
            cancel_initiated=row.cancel_initiated,
        )

    @staticmethod
    def _row_to_batch_request(row: BatchRequestTable) -> BatchRequest:
        return BatchRequest(
            id=row.id,
            batch_id=row.batch_id,
            custom_id=row.custom_id,
            position=row.position,
            params=row.params,
            status=row.status,
            attempts=row.attempts,
            response=row.response,
            error=row.error,
        )
