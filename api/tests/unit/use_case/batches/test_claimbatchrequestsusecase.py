from datetime import UTC, datetime, timedelta
from unittest.mock import create_autospec

import pytest

from api.domain.batch import BatchRepository
from api.domain.batch.entities import BatchRequestError, BatchRequestStatus, ClaimedBatchRequest
from api.domain.key import KeyRepository
from api.domain.user import AuthenticatedUserQuery
from api.tests.unit.use_case.factories import AuthenticatedUserFactory, KeyFactory
from api.use_cases.batches import BatchRequestClaim, ClaimBatchRequestsCommand, ClaimBatchRequestsUseCase

COMMAND = ClaimBatchRequestsCommand(limit=4, lease_seconds=600)


@pytest.fixture
def claimed_request():
    return ClaimedBatchRequest(
        id=11,
        batch_id=3,
        custom_id="request-1",
        position=0,
        params={"model": "m"},
        status=BatchRequestStatus.PROCESSING,
        attempts=0,
        claim_id="claim-1",
        user_id=1,
        key_id=2,
    )


@pytest.fixture
def use_case(claimed_request):
    use_case = ClaimBatchRequestsUseCase(
        batch_repository=create_autospec(BatchRepository, instance=True, spec_set=True),
        key_repository=create_autospec(KeyRepository, instance=True, spec_set=True),
        authenticated_user_query=create_autospec(AuthenticatedUserQuery, instance=True, spec_set=True),
    )
    use_case.batch_repository.claim_requests.return_value = [claimed_request]
    return use_case


class TestClaimBatchRequestsUseCase:
    @pytest.mark.asyncio
    async def test_should_return_the_claimed_requests_with_their_owner(self, use_case, claimed_request):
        # Arrange
        key = KeyFactory(id=2, user_id=1)
        user = AuthenticatedUserFactory(id=1)
        use_case.key_repository.get_key_by_id.return_value = key
        use_case.authenticated_user_query.get_user_by_id.return_value = user

        # Act
        result = await use_case.execute(COMMAND)

        # Assert
        assert result.claims == [BatchRequestClaim(request=claimed_request, authenticated_user=user, authenticated_key=key)]
        use_case.batch_repository.claim_requests.assert_awaited_once_with(limit=4, lease_seconds=600)
        use_case.batch_repository.complete_request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_should_complete_as_errored_a_request_whose_key_was_revoked(self, use_case):
        # Arrange
        use_case.key_repository.get_key_by_id.return_value = KeyFactory(id=2, revoked=True)

        # Act
        result = await use_case.execute(COMMAND)

        # Assert
        assert result.claims == []
        use_case.batch_repository.complete_request.assert_awaited_once_with(
            claim_id="claim-1",
            status=BatchRequestStatus.ERRORED,
            response=None,
            error=BatchRequestError(type="InvalidKeyError", message="The API key that created the batch is revoked or expired."),
        )
        use_case.authenticated_user_query.get_user_by_id.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_should_complete_as_errored_a_request_whose_user_expired(self, use_case):
        # Arrange
        use_case.key_repository.get_key_by_id.return_value = KeyFactory(id=2)
        use_case.authenticated_user_query.get_user_by_id.return_value = AuthenticatedUserFactory(
            id=1, expires=datetime.now(tz=UTC) - timedelta(days=1)
        )

        # Act
        result = await use_case.execute(COMMAND)

        # Assert
        assert result.claims == []
        assert use_case.batch_repository.complete_request.await_args.kwargs["error"].type == "AccountExpiredError"
