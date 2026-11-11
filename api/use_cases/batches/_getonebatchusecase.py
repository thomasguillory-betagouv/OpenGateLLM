from dataclasses import dataclass

from api.domain.batch import BatchRepository
from api.domain.batch.entities import Batch
from api.domain.batch.errors import BatchNotFoundError


@dataclass
class GetOneBatchCommand:
    batch_id: int
    user_id: int


@dataclass
class GetOneBatchUseCaseSuccess:
    batch: Batch


type GetOneBatchUseCaseResult = GetOneBatchUseCaseSuccess | BatchNotFoundError


class GetOneBatchUseCase:
    def __init__(self, batch_repository: BatchRepository):
        self.batch_repository = batch_repository

    async def execute(self, command: GetOneBatchCommand) -> GetOneBatchUseCaseResult:
        result = await self.batch_repository.get_batch(batch_id=command.batch_id, user_id=command.user_id)
        match result:
            case Batch() as batch:
                return GetOneBatchUseCaseSuccess(batch=batch)
            case BatchNotFoundError() as error:
                return error
