from dataclasses import dataclass

from api.domain.batch import BatchRepository


@dataclass
class ReconcileBatchesUseCaseSuccess:
    released_requests: int
    expired_requests: int
    ended_batches: int


class ReconcileBatchesUseCase:
    """Housekeeping the worker runs periodically: no request event marks these transitions."""

    def __init__(self, batch_repository: BatchRepository):
        self.batch_repository = batch_repository

    async def execute(self) -> ReconcileBatchesUseCaseSuccess:
        released_requests = await self.batch_repository.release_expired_claims()
        expired_requests = await self.batch_repository.expire_overdue_requests()
        ended_batches = await self.batch_repository.end_finished_batches()

        return ReconcileBatchesUseCaseSuccess(released_requests=released_requests, expired_requests=expired_requests, ended_batches=ended_batches)
