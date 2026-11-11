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
        """Mark up to `limit` pending requests of batches in progress as processing and return them, least advanced batch
        first: a small batch submitted after a large one does not wait for it.

        A claim lasts `lease_seconds` unless extended (see extend_claims). A claim that expires goes back to the queue (see
        release_expired_claims), so a dead worker loses no request."""
        pass

    @abstractmethod
    async def extend_claims(self, claim_ids: list[str], lease_seconds: int) -> None:
        pass

    @abstractmethod
    async def complete_request(self, claim_id: str, status: BatchRequestStatus, response: dict | None, error: BatchRequestError | None) -> None:
        """Store the outcome of a claimed request. Does nothing if the claim has expired in the meantime."""
        pass

    @abstractmethod
    async def requeue_request(self, claim_id: str, delay_seconds: float, count_attempt: bool) -> None:
        """Put a claimed request back in the queue. Does nothing if the claim has expired in the meantime."""
        pass

    @abstractmethod
    async def release_expired_claims(self) -> int:
        """Put the requests whose claim expired back in the queue, counting an attempt: a request that kills its worker
        must not be retried forever."""
        pass

    @abstractmethod
    async def cancel_requests_of_canceling_batches(self) -> int:
        """Cancel the requests that went back to the queue after their batch was canceled."""
        pass

    @abstractmethod
    async def expire_overdue_requests(self) -> int:
        pass

    @abstractmethod
    async def end_finished_batches(self) -> int:
        pass
