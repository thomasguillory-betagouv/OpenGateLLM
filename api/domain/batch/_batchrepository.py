from abc import ABC, abstractmethod
from datetime import datetime

from api.domain.batch.entities import Batch, BatchPage, BatchRequest, BatchRequestError, BatchRequestStatus, BatchRequestToCreate, ClaimedBatchRequest
from api.domain.batch.errors import BatchNotFoundError


class BatchRepository(ABC):
    @abstractmethod
    async def create_batch(self, user_id: int, key_id: int, expires: datetime, requests: list[BatchRequestToCreate]) -> Batch:
        pass

    @abstractmethod
    async def get_batch(self, batch_id: int, user_id: int) -> Batch | BatchNotFoundError:
        pass

    @abstractmethod
    async def get_batches_page(self, user_id: int, offset: int, limit: int) -> BatchPage:
        pass

    @abstractmethod
    async def cancel_batch(self, batch_id: int, user_id: int) -> Batch | BatchNotFoundError:
        """Mark the batch as canceling and cancel its pending requests. Requests already processing run to the end."""
        pass

    @abstractmethod
    async def get_batch_requests(self, batch_id: int, after_position: int, limit: int) -> list[BatchRequest]:
        """Return the requests of the batch in submission order, starting after `after_position` (-1 for the first page)."""
        pass

    @abstractmethod
    async def claim_requests(self, limit: int, lease_seconds: int) -> list[ClaimedBatchRequest]:
        """Mark up to `limit` pending requests as processing and return them. A request whose lease expires before it is
        completed or requeued goes back to the queue (see release_expired_claims), so a dead worker loses no request.

        Requests are claimed least advanced batch first: a small batch submitted after a large one does not wait for it."""
        pass

    @abstractmethod
    async def complete_request(self, request_id: int, status: BatchRequestStatus, response: dict | None, error: BatchRequestError | None) -> None:
        pass

    @abstractmethod
    async def requeue_request(self, request_id: int, delay_seconds: float, count_attempt: bool) -> None:
        pass

    @abstractmethod
    async def release_expired_claims(self) -> int:
        pass

    @abstractmethod
    async def expire_overdue_requests(self) -> int:
        pass

    @abstractmethod
    async def end_finished_batches(self) -> int:
        pass
