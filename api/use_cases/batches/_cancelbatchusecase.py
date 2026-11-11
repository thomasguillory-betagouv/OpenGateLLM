from dataclasses import dataclass

from api.domain.batch import BatchRepository
from api.domain.batch.entities import Batch
from api.domain.batch.errors import BatchAlreadyEndedError, BatchNotFoundError


@dataclass
class CancelBatchCommand:
    batch_id: int
    user_id: int


@dataclass
class CancelBatchUseCaseSuccess:
    batch: Batch


type CancelBatchUseCaseResult = CancelBatchUseCaseSuccess | BatchNotFoundError | BatchAlreadyEndedError


class CancelBatchUseCase:
    def __init__(self, batch_repository: BatchRepository):
        self.batch_repository = batch_repository

    async def execute(self, command: CancelBatchCommand) -> CancelBatchUseCaseResult:
        result = await self.batch_repository.get_batch(batch_id=command.batch_id, user_id=command.user_id)
        match result:
            case Batch() as batch:
                pass
            case BatchNotFoundError() as error:
                return error

        if batch.has_ended:
            return BatchAlreadyEndedError(id=batch.id)

        result = await self.batch_repository.cancel_batch(batch_id=batch.id, user_id=command.user_id)
        match result:
            case Batch() as canceled_batch:
                return CancelBatchUseCaseSuccess(batch=canceled_batch)
            case BatchNotFoundError() as error:
                return error
