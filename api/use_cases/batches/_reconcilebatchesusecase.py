from dataclasses import dataclass

from api.domain.batch import BatchRepository


@dataclass
class ReconcileBatchesUseCaseSuccess:
    released_requests: int
    canceled_requests: int
    expired_requests: int
    ended_batches: int


class ReconcileBatchesUseCase:
    """Housekeeping the worker runs periodically: no request event marks these transitions. The order matters: a claim
    released back to the queue may belong to a canceled or overdue batch, and a batch ends once nothing is left pending."""

    def __init__(self, batch_repository: BatchRepository):
        self.batch_repository = batch_repository

    async def execute(self) -> ReconcileBatchesUseCaseSuccess:
        released_requests = await self.batch_repository.release_expired_claims()
        canceled_requests = await self.batch_repository.cancel_requests_of_canceling_batches()
        expired_requests = await self.batch_repository.expire_overdue_requests()
        ended_batches = await self.batch_repository.end_finished_batches()

        return ReconcileBatchesUseCaseSuccess(
            released_requests=released_requests,
            canceled_requests=canceled_requests,
            expired_requests=expired_requests,
            ended_batches=ended_batches,
        )
