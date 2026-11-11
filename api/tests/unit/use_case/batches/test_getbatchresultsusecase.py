from datetime import UTC, datetime
from unittest.mock import create_autospec

import pytest

from api.domain.batch import BatchRepository
from api.domain.batch.entities import Batch, BatchProcessingStatus, BatchRequest, BatchRequestCounts, BatchRequestStatus
from api.domain.batch.errors import BatchNotEndedError, BatchNotFoundError
from api.use_cases.batches import GetBatchResultsCommand, GetBatchResultsUseCase, GetBatchResultsUseCaseSuccess

COMMAND = GetBatchResultsCommand(batch_id=3, user_id=1)


def _batch(processing_status: BatchProcessingStatus) -> Batch:
    now = datetime.now(tz=UTC)
    return Batch(id=3, user_id=1, key_id=2, processing_status=processing_status, request_counts=BatchRequestCounts(), created=now, expires=now)


def _request(position: int) -> BatchRequest:
    return BatchRequest(
        id=position + 100, batch_id=3, custom_id=f"request-{position}", position=position, params={}, status=BatchRequestStatus.SUCCEEDED, attempts=0
    )


@pytest.fixture
def use_case():
    return GetBatchResultsUseCase(batch_repository=create_autospec(BatchRepository, instance=True, spec_set=True))


class TestGetBatchResultsUseCase:
    @pytest.mark.asyncio
    async def test_should_read_the_results_page_by_page(self, use_case):
        # Arrange
        use_case.PAGE_SIZE = 2
        use_case.batch_repository.get_batch.return_value = _batch(BatchProcessingStatus.ENDED)
        use_case.batch_repository.get_batch_requests.side_effect = [[_request(0), _request(1)], [_request(2)]]

        # Act
        result = await use_case.execute(COMMAND)
        results = [request async for request in result.results]

        # Assert
        assert isinstance(result, GetBatchResultsUseCaseSuccess)
        assert [request.position for request in results] == [0, 1, 2]
        assert [call.kwargs["after_position"] for call in use_case.batch_repository.get_batch_requests.await_args_list] == [-1, 1]

    @pytest.mark.asyncio
    async def test_should_return_not_found_for_an_unknown_batch(self, use_case):
        # Arrange
        use_case.batch_repository.get_batch.return_value = BatchNotFoundError(id=3)

        # Act
        result = await use_case.execute(COMMAND)

        # Assert
        assert result == BatchNotFoundError(id=3)

    @pytest.mark.asyncio
    async def test_should_refuse_the_results_of_a_batch_not_ended(self, use_case):
        # Arrange
        use_case.batch_repository.get_batch.return_value = _batch(BatchProcessingStatus.IN_PROGRESS)

        # Act
        result = await use_case.execute(COMMAND)

        # Assert
        assert result == BatchNotEndedError(id=3)
        use_case.batch_repository.get_batch_requests.assert_not_awaited()
