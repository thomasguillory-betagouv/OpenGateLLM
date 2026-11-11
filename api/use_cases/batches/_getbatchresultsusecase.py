from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import ClassVar

from api.domain.batch import BatchRepository
from api.domain.batch.entities import Batch, BatchRequest
from api.domain.batch.errors import BatchNotEndedError, BatchNotFoundError


@dataclass
class GetBatchResultsCommand:
    batch_id: int
    user_id: int


@dataclass
class GetBatchResultsUseCaseSuccess:
    results: AsyncGenerator[BatchRequest]


type GetBatchResultsUseCaseResult = GetBatchResultsUseCaseSuccess | BatchNotFoundError | BatchNotEndedError


class GetBatchResultsUseCase:
    PAGE_SIZE: ClassVar[int] = 500

    def __init__(self, batch_repository: BatchRepository):
        self.batch_repository = batch_repository

    async def execute(self, command: GetBatchResultsCommand) -> GetBatchResultsUseCaseResult:
        result = await self.batch_repository.get_batch(batch_id=command.batch_id, user_id=command.user_id)
        match result:
            case Batch() as batch:
                pass
            case BatchNotFoundError() as error:
                return error

        if not batch.has_ended:
            return BatchNotEndedError(id=batch.id)

        # lazy: a batch holds up to thousands of results, read page by page while the response is sent
        return GetBatchResultsUseCaseSuccess(results=self._read_results(batch_id=batch.id))

    async def _read_results(self, batch_id: int) -> AsyncGenerator[BatchRequest]:
        after_position = -1
        while True:
            page = await self.batch_repository.get_batch_requests(batch_id=batch_id, after_position=after_position, limit=self.PAGE_SIZE)
            for request in page:
                yield request
            if len(page) < self.PAGE_SIZE:
                return
            after_position = page[-1].position
