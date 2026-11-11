from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select

from api.domain.batch.entities import (
    Batch,
    BatchProcessingStatus,
    BatchRequestCounts,
    BatchRequestError,
    BatchRequestStatus,
    BatchRequestToCreate,
    ClaimedBatchRequest,
)
from api.domain.batch.errors import BatchNotFoundError
from api.domain.usage.entities import Usage, UsageRecord
from api.infrastructure.postgres import PostgresBatchRepository
from api.infrastructure.postgres.models import BatchRequest as BatchRequestTable
from api.infrastructure.postgres.models import Usage as UsageTable
from api.tests.integration.factories.sql import BatchRequestSQLFactory, BatchSQLFactory, KeySQLFactory

LEASE_SECONDS = 600


@pytest.fixture
def repository(db_session):
    return PostgresBatchRepository(postgres_session=db_session)


async def _stored_request(db_session, request_id: int) -> BatchRequestTable:
    # the repository writes with SQL UPDATEs, which leave the objects the factories created untouched
    return await db_session.scalar(select(BatchRequestTable).where(BatchRequestTable.id == request_id).execution_options(populate_existing=True))


@pytest.mark.asyncio(loop_scope="session")
class TestCreateBatch:
    async def test_creates_the_batch_and_its_requests_in_submission_order(self, repository, db_session):
        # Arrange
        key = KeySQLFactory()
        await db_session.flush()
        expires = datetime.now(tz=UTC) + timedelta(hours=24)
        requests = [BatchRequestToCreate(custom_id=f"request-{index}", params={"model": "my-model", "index": index}) for index in range(3)]

        # Act
        batch = await repository.create_batch(user_id=key.user_id, key_id=key.id, expires=expires, requests=requests)

        # Assert
        assert batch.processing_status == BatchProcessingStatus.IN_PROGRESS
        assert batch.request_counts == BatchRequestCounts(processing=3)
        stored = await repository.get_batch_requests(batch_id=batch.id, after_position=-1, limit=10)
        assert [(request.custom_id, request.position, request.status) for request in stored] == [
            ("request-0", 0, BatchRequestStatus.PENDING),
            ("request-1", 1, BatchRequestStatus.PENDING),
            ("request-2", 2, BatchRequestStatus.PENDING),
        ]


@pytest.mark.asyncio(loop_scope="session")
class TestGetBatch:
    async def test_returns_the_batch_with_its_request_counts(self, repository, db_session):
        # Arrange
        batch = BatchSQLFactory()
        BatchRequestSQLFactory(batch=batch, status=BatchRequestStatus.PENDING)
        BatchRequestSQLFactory(batch=batch, status=BatchRequestStatus.PROCESSING)
        BatchRequestSQLFactory(batch=batch, status=BatchRequestStatus.SUCCEEDED)
        BatchRequestSQLFactory(batch=batch, status=BatchRequestStatus.ERRORED)
        await db_session.flush()

        # Act
        result = await repository.get_batch(batch_id=batch.id, user_id=batch.user_id)

        # Assert
        assert isinstance(result, Batch)
        assert result.request_counts == BatchRequestCounts(processing=2, succeeded=1, errored=1)

    async def test_returns_not_found_for_the_batch_of_another_user(self, repository, db_session):
        # Arrange
        batch = BatchSQLFactory()
        other_key = KeySQLFactory()
        await db_session.flush()

        # Act
        result = await repository.get_batch(batch_id=batch.id, user_id=other_key.user_id)

        # Assert
        assert result == BatchNotFoundError(id=batch.id)


@pytest.mark.asyncio(loop_scope="session")
class TestGetBatchesPage:
    async def test_returns_only_the_batches_of_the_user_newest_first(self, repository, db_session):
        # Arrange
        key = KeySQLFactory()
        older = BatchSQLFactory(key=key)
        newer = BatchSQLFactory(key=key)
        BatchSQLFactory()
        await db_session.flush()

        # Act
        page = await repository.get_batches_page(user_id=key.user_id, offset=0, limit=10)

        # Assert
        assert page.total == 2
        assert [batch.id for batch in page.data] == [newer.id, older.id]


@pytest.mark.asyncio(loop_scope="session")
class TestCancelBatch:
    async def test_cancels_the_pending_requests_and_lets_the_processing_ones_finish(self, repository, db_session):
        # Arrange
        batch = BatchSQLFactory()
        pending = BatchRequestSQLFactory(batch=batch, status=BatchRequestStatus.PENDING)
        processing = BatchRequestSQLFactory(batch=batch, status=BatchRequestStatus.PROCESSING)
        await db_session.flush()

        # Act
        result = await repository.cancel_batch(batch_id=batch.id, user_id=batch.user_id)

        # Assert
        assert result.processing_status == BatchProcessingStatus.CANCELING
        assert result.cancel_initiated is not None
        assert (await _stored_request(db_session, pending.id)).status == BatchRequestStatus.CANCELED
        assert (await _stored_request(db_session, processing.id)).status == BatchRequestStatus.PROCESSING


@pytest.mark.asyncio(loop_scope="session")
class TestClaimRequests:
    async def test_claims_the_least_advanced_batch_first(self, repository, db_session):
        # Arrange
        advanced_batch = BatchSQLFactory()
        BatchRequestSQLFactory(batch=advanced_batch, position=0, status=BatchRequestStatus.SUCCEEDED)
        BatchRequestSQLFactory(batch=advanced_batch, position=1, status=BatchRequestStatus.SUCCEEDED)
        advanced_next = BatchRequestSQLFactory(batch=advanced_batch, position=2)
        new_batch = BatchSQLFactory()
        new_first = BatchRequestSQLFactory(batch=new_batch, position=0)
        new_second = BatchRequestSQLFactory(batch=new_batch, position=1)
        await db_session.flush()

        # Act
        claimed = await repository.claim_requests(limit=2, lease_seconds=LEASE_SECONDS)

        # Assert
        assert [request.id for request in claimed] == [new_first.id, new_second.id]
        assert (await _stored_request(db_session, advanced_next.id)).status == BatchRequestStatus.PENDING

    async def test_marks_the_claimed_request_as_processing_and_returns_its_owner(self, repository, db_session):
        # Arrange
        batch = BatchSQLFactory()
        request = BatchRequestSQLFactory(batch=batch)
        await db_session.flush()
        # read before the Act: refreshing the request afterwards would unload its batch, and the lazy load cannot run here
        expected = {
            "id": request.id,
            "batch_id": batch.id,
            "custom_id": request.custom_id,
            "position": request.position,
            "params": request.params,
            "user_id": batch.user_id,
            "key_id": batch.key_id,
        }

        # Act
        claimed = await repository.claim_requests(limit=10, lease_seconds=LEASE_SECONDS)

        # Assert
        stored = await _stored_request(db_session, expected["id"])
        assert claimed == [ClaimedBatchRequest(**expected, status=BatchRequestStatus.PROCESSING, attempts=0, claim_id=stored.claim_id)]
        assert stored.status == BatchRequestStatus.PROCESSING
        assert stored.claim_id is not None
        assert stored.locked_until is not None

    async def test_skips_requests_not_yet_available_and_requests_of_expired_or_canceling_batches(self, repository, db_session):
        # Arrange
        BatchRequestSQLFactory(available_at=datetime.now(tz=UTC) + timedelta(hours=1))
        BatchRequestSQLFactory(batch=BatchSQLFactory(expires=datetime.now(tz=UTC) - timedelta(hours=1)))
        BatchRequestSQLFactory(batch=BatchSQLFactory(processing_status=BatchProcessingStatus.CANCELING))
        await db_session.flush()

        # Act
        claimed = await repository.claim_requests(limit=10, lease_seconds=LEASE_SECONDS)

        # Assert
        assert claimed == []


@pytest.mark.asyncio(loop_scope="session")
class TestCompleteRequest:
    async def test_stores_the_result_of_a_claimed_request(self, repository, db_session):
        # Arrange
        request = BatchRequestSQLFactory(status=BatchRequestStatus.PROCESSING, claim_id="claim-1")
        await db_session.flush()
        error = BatchRequestError(type="StatusCodeModelError", message="Bad request", status_code=400)

        # Act
        await repository.complete_request(claim_id="claim-1", status=BatchRequestStatus.ERRORED, response=None, error=error)

        # Assert
        stored = await _stored_request(db_session, request.id)
        assert stored.status == BatchRequestStatus.ERRORED
        assert stored.error == error.model_dump()
        assert stored.claim_id is None
        assert stored.completed is not None

    async def test_ignores_the_outcome_of_a_claim_that_expired_and_was_claimed_again(self, repository, db_session):
        # Arrange
        request = BatchRequestSQLFactory(status=BatchRequestStatus.PROCESSING, claim_id="second-claim")
        await db_session.flush()

        # Act
        await repository.complete_request(claim_id="first-claim", status=BatchRequestStatus.SUCCEEDED, response={"id": "stale"}, error=None)

        # Assert
        stored = await _stored_request(db_session, request.id)
        assert stored.status == BatchRequestStatus.PROCESSING
        assert stored.response is None


def _usage_record(user_id: int, request_id: str = "batch-request-1") -> UsageRecord:
    return UsageRecord(
        request_id=request_id,
        endpoint="/v1/chat/completions/batches",
        created=datetime.now(tz=UTC),
        user_id=user_id,
        latency=500,
        usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30, cost=0.5),
    )


async def _usage_rows(db_session, request_id: str) -> list[UsageTable]:
    return list(await db_session.scalars(select(UsageTable).where(UsageTable.request_id == request_id)))


@pytest.mark.asyncio(loop_scope="session")
class TestCompleteRequestAndSaveUsage:
    async def test_stores_the_result_and_its_usage_row_together(self, repository, db_session):
        # Arrange
        batch = BatchSQLFactory()
        request = BatchRequestSQLFactory(batch=batch, status=BatchRequestStatus.PROCESSING, claim_id="claim-1")
        await db_session.flush()
        request_id, user_id = request.id, batch.user_id

        # Act
        await repository.complete_request_and_save_usage(claim_id="claim-1", response={"id": "chatcmpl-1"}, usage_record=_usage_record(user_id))

        # Assert
        stored = await _stored_request(db_session, request_id)
        assert (stored.status, stored.response, stored.claim_id) == (BatchRequestStatus.SUCCEEDED, {"id": "chatcmpl-1"}, None)
        [usage_row] = await _usage_rows(db_session, "batch-request-1")
        assert (usage_row.user_id, usage_row.endpoint, usage_row.total_tokens, usage_row.cost, usage_row.status) == (
            user_id,
            "/v1/chat/completions/batches",
            30,
            0.5,
            200,
        )

    async def test_bills_nothing_for_a_claim_that_expired_and_was_claimed_again(self, repository, db_session):
        # Arrange
        batch = BatchSQLFactory()
        request = BatchRequestSQLFactory(batch=batch, status=BatchRequestStatus.PROCESSING, claim_id="second-claim")
        await db_session.flush()
        request_id, user_id = request.id, batch.user_id
        usage_rows_before = await db_session.scalar(select(func.count()).select_from(UsageTable))

        # Act
        await repository.complete_request_and_save_usage(claim_id="first-claim", response={"id": "stale"}, usage_record=_usage_record(user_id))

        # Assert
        assert (await _stored_request(db_session, request_id)).status == BatchRequestStatus.PROCESSING
        assert await db_session.scalar(select(func.count()).select_from(UsageTable)) == usage_rows_before


@pytest.mark.asyncio(loop_scope="session")
class TestRequeueRequest:
    async def test_puts_the_request_back_in_the_queue_after_the_delay(self, repository, db_session):
        # Arrange
        request = BatchRequestSQLFactory(status=BatchRequestStatus.PROCESSING, attempts=1, claim_id="claim-1")
        await db_session.flush()

        # Act
        await repository.requeue_request(claim_id="claim-1", delay_seconds=60, count_attempt=True)

        # Assert
        stored = await _stored_request(db_session, request.id)
        assert stored.status == BatchRequestStatus.PENDING
        assert stored.attempts == 2
        assert await repository.claim_requests(limit=10, lease_seconds=LEASE_SECONDS) == []


@pytest.mark.asyncio(loop_scope="session")
class TestExtendClaims:
    async def test_extends_the_lease_of_the_given_claims(self, repository, db_session):
        # Arrange
        almost_expired = datetime.now(tz=UTC) - timedelta(hours=2)
        extended = BatchRequestSQLFactory(status=BatchRequestStatus.PROCESSING, claim_id="claim-1", locked_until=almost_expired)
        other = BatchRequestSQLFactory(status=BatchRequestStatus.PROCESSING, claim_id="claim-2", locked_until=almost_expired)
        await db_session.flush()

        # Act
        await repository.extend_claims(claim_ids=["claim-1"], lease_seconds=60)

        # Assert
        assert (await _stored_request(db_session, extended.id)).locked_until > almost_expired
        assert (await _stored_request(db_session, other.id)).locked_until == almost_expired


@pytest.mark.asyncio(loop_scope="session")
class TestReleaseExpiredClaims:
    async def test_returns_the_requests_of_a_dead_worker_to_the_queue_counting_an_attempt(self, repository, db_session):
        # Arrange
        abandoned = BatchRequestSQLFactory(
            status=BatchRequestStatus.PROCESSING, claim_id="claim-1", attempts=1, locked_until=datetime.now(tz=UTC) - timedelta(hours=1)
        )
        in_progress = BatchRequestSQLFactory(status=BatchRequestStatus.PROCESSING, locked_until=datetime.now(tz=UTC) + timedelta(hours=1))
        await db_session.flush()

        # Act
        released = await repository.release_expired_claims()

        # Assert
        assert released == 1
        stored = await _stored_request(db_session, abandoned.id)
        assert (stored.status, stored.attempts, stored.claim_id) == (BatchRequestStatus.PENDING, 2, None)
        assert (await _stored_request(db_session, in_progress.id)).status == BatchRequestStatus.PROCESSING


@pytest.mark.asyncio(loop_scope="session")
class TestCancelRequestsOfCancelingBatches:
    async def test_cancels_the_requests_that_went_back_to_the_queue_after_the_cancellation(self, repository, db_session):
        # Arrange
        requeued = BatchRequestSQLFactory(batch=BatchSQLFactory(processing_status=BatchProcessingStatus.CANCELING))
        in_progress_batch_request = BatchRequestSQLFactory()
        await db_session.flush()

        # Act
        canceled = await repository.cancel_requests_of_canceling_batches()

        # Assert
        assert canceled == 1
        assert (await _stored_request(db_session, requeued.id)).status == BatchRequestStatus.CANCELED
        assert (await _stored_request(db_session, in_progress_batch_request.id)).status == BatchRequestStatus.PENDING


@pytest.mark.asyncio(loop_scope="session")
class TestExpireOverdueRequests:
    async def test_expires_the_pending_requests_of_overdue_batches(self, repository, db_session):
        # Arrange
        overdue_batch = BatchSQLFactory(expires=datetime.now(tz=UTC) - timedelta(hours=1))
        overdue = BatchRequestSQLFactory(batch=overdue_batch)
        on_time = BatchRequestSQLFactory()
        await db_session.flush()

        # Act
        expired = await repository.expire_overdue_requests()

        # Assert
        assert expired == 1
        assert (await _stored_request(db_session, overdue.id)).status == BatchRequestStatus.EXPIRED
        assert (await _stored_request(db_session, on_time.id)).status == BatchRequestStatus.PENDING


@pytest.mark.asyncio(loop_scope="session")
class TestEndFinishedBatches:
    async def test_ends_the_batches_whose_requests_are_all_finished(self, repository, db_session):
        # Arrange
        finished_batch = BatchSQLFactory(processing_status=BatchProcessingStatus.CANCELING)
        BatchRequestSQLFactory(batch=finished_batch, status=BatchRequestStatus.SUCCEEDED)
        BatchRequestSQLFactory(batch=finished_batch, status=BatchRequestStatus.CANCELED)
        unfinished_batch = BatchSQLFactory()
        BatchRequestSQLFactory(batch=unfinished_batch, status=BatchRequestStatus.SUCCEEDED)
        BatchRequestSQLFactory(batch=unfinished_batch, status=BatchRequestStatus.PROCESSING)
        await db_session.flush()

        # Act
        ended = await repository.end_finished_batches()

        # Assert
        assert ended == 1
        finished = await repository.get_batch(batch_id=finished_batch.id, user_id=finished_batch.user_id)
        unfinished = await repository.get_batch(batch_id=unfinished_batch.id, user_id=unfinished_batch.user_id)
        assert finished.processing_status == BatchProcessingStatus.ENDED
        assert finished.ended is not None
        assert unfinished.processing_status == BatchProcessingStatus.IN_PROGRESS
