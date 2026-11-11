from datetime import UTC, datetime
from unittest.mock import create_autospec

import pytest

from api.domain.batch import BatchRepository
from api.domain.batch.entities import Batch, BatchProcessingStatus, BatchRequestCounts
from api.domain.batch.errors import BatchAlreadyEndedError, BatchNotFoundError
from api.use_cases.batches import CancelBatchCommand, CancelBatchUseCase, CancelBatchUseCaseSuccess

COMMAND = CancelBatchCommand(batch_id=3, user_id=1)


def _batch(processing_status: BatchProcessingStatus) -> Batch:
    now = datetime.now(tz=UTC)
    return Batch(id=3, user_id=1, key_id=2, processing_status=processing_status, request_counts=BatchRequestCounts(), created=now, expires=now)


@pytest.fixture
def use_case():
    return CancelBatchUseCase(batch_repository=create_autospec(BatchRepository, instance=True, spec_set=True))


class TestCancelBatchUseCase:
    @pytest.mark.asyncio
    async def test_should_cancel_a_batch_in_progress(self, use_case):
        # Arrange
        canceling = _batch(BatchProcessingStatus.CANCELING)
        use_case.batch_repository.get_batch.return_value = _batch(BatchProcessingStatus.IN_PROGRESS)
        use_case.batch_repository.cancel_batch.return_value = canceling

        # Act
        result = await use_case.execute(COMMAND)

        # Assert
        assert result == CancelBatchUseCaseSuccess(batch=canceling)
        use_case.batch_repository.cancel_batch.assert_awaited_once_with(batch_id=3, user_id=1)

    @pytest.mark.asyncio
    async def test_should_return_not_found_for_an_unknown_batch(self, use_case):
        # Arrange
        use_case.batch_repository.get_batch.return_value = BatchNotFoundError(id=3)

        # Act
        result = await use_case.execute(COMMAND)

        # Assert
        assert result == BatchNotFoundError(id=3)
        use_case.batch_repository.cancel_batch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_should_refuse_to_cancel_an_ended_batch(self, use_case):
        # Arrange
        use_case.batch_repository.get_batch.return_value = _batch(BatchProcessingStatus.ENDED)

        # Act
        result = await use_case.execute(COMMAND)

        # Assert
        assert result == BatchAlreadyEndedError(id=3)
        use_case.batch_repository.cancel_batch.assert_not_awaited()
