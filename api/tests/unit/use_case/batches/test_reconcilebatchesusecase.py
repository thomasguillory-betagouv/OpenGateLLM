from unittest.mock import Mock, call, create_autospec

import pytest

from api.domain.batch import BatchRepository
from api.use_cases.batches import ReconcileBatchesUseCase, ReconcileBatchesUseCaseSuccess


class TestReconcileBatchesUseCase:
    @pytest.mark.asyncio
    async def test_should_release_cancel_and_expire_requests_before_ending_the_finished_batches(self):
        # Arrange
        mock_batch_repository = create_autospec(BatchRepository, instance=True, spec_set=True)
        mock_batch_repository.release_expired_claims.return_value = 1
        mock_batch_repository.cancel_requests_of_canceling_batches.return_value = 2
        mock_batch_repository.expire_overdue_requests.return_value = 3
        mock_batch_repository.end_finished_batches.return_value = 4
        mock_calls = Mock()
        mock_calls.attach_mock(mock_batch_repository.release_expired_claims, "release_expired_claims")
        mock_calls.attach_mock(mock_batch_repository.cancel_requests_of_canceling_batches, "cancel_requests_of_canceling_batches")
        mock_calls.attach_mock(mock_batch_repository.expire_overdue_requests, "expire_overdue_requests")
        mock_calls.attach_mock(mock_batch_repository.end_finished_batches, "end_finished_batches")

        # Act
        result = await ReconcileBatchesUseCase(batch_repository=mock_batch_repository).execute()

        # Assert
        assert result == ReconcileBatchesUseCaseSuccess(released_requests=1, canceled_requests=2, expired_requests=3, ended_batches=4)
        assert mock_calls.mock_calls == [
            call.release_expired_claims(),
            call.cancel_requests_of_canceling_batches(),
            call.expire_overdue_requests(),
            call.end_finished_batches(),
        ]
