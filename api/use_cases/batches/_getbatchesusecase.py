from dataclasses import dataclass

from api.domain.batch import BatchRepository
from api.domain.batch.entities import BatchPage


@dataclass
class GetBatchesCommand:
    user_id: int
    offset: int
    limit: int


@dataclass
class GetBatchesUseCaseSuccess:
    batch_page: BatchPage


class GetBatchesUseCase:
    def __init__(self, batch_repository: BatchRepository):
        self.batch_repository = batch_repository

    async def execute(self, command: GetBatchesCommand) -> GetBatchesUseCaseSuccess:
        batch_page = await self.batch_repository.get_batches_page(user_id=command.user_id, offset=command.offset, limit=command.limit)
        return GetBatchesUseCaseSuccess(batch_page=batch_page)
