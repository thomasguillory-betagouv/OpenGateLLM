from unittest.mock import create_autospec

import pytest

from api.domain.batch import BatchRepository
from api.use_cases.batches import ReconcileBatchesUseCase, ReconcileBatchesUseCaseSuccess


class TestReconcileBatchesUseCase:
    @pytest.mark.asyncio
    async def test_should_release_expired_claims_expire_overdue_requests_and_end_finished_batches(self):
        # Arrange
        batch_repository = create_autospec(BatchRepository, instance=True, spec_set=True)
        batch_repository.release_expired_claims.return_value = 1
        batch_repository.expire_overdue_requests.return_value = 2
        batch_repository.end_finished_batches.return_value = 3

        # Act
        result = await ReconcileBatchesUseCase(batch_repository=batch_repository).execute()

        # Assert
        assert result == ReconcileBatchesUseCaseSuccess(released_requests=1, expired_requests=2, ended_batches=3)
