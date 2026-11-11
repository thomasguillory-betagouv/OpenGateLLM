from dataclasses import dataclass
from datetime import UTC, datetime

from api.domain.batch import BatchRepository
from api.domain.batch.entities import BatchRequestError, BatchRequestStatus, ClaimedBatchRequest
from api.domain.key import KeyRepository
from api.domain.key.entities import Key
from api.domain.user import AuthenticatedUserQuery
from api.domain.user.views import AuthenticatedUserView


@dataclass
class ClaimBatchRequestsCommand:
    limit: int
    lease_seconds: int


@dataclass
class BatchRequestClaim:
    request: ClaimedBatchRequest
    authenticated_user: AuthenticatedUserView
    authenticated_key: Key


@dataclass
class ClaimBatchRequestsUseCaseSuccess:
    claims: list[BatchRequestClaim]


class ClaimBatchRequestsUseCase:
    def __init__(self, batch_repository: BatchRepository, key_repository: KeyRepository, authenticated_user_query: AuthenticatedUserQuery):
        self.batch_repository = batch_repository
        self.key_repository = key_repository
        self.authenticated_user_query = authenticated_user_query

    async def execute(self, command: ClaimBatchRequestsCommand) -> ClaimBatchRequestsUseCaseSuccess:
        requests = await self.batch_repository.claim_requests(limit=command.limit, lease_seconds=command.lease_seconds)

        claims = []
        for request in requests:
            # the owner is checked again: since the batch was created, its key may have been revoked or its user may have expired
            result = await self.key_repository.get_key_by_id(key_id=request.key_id)
            match result:
                case Key() as key if self._is_usable(key):
                    pass
                case _:
                    error = BatchRequestError(type="InvalidKeyError", message="The API key that created the batch is revoked or expired.")
                    await self.batch_repository.complete_request(
                        claim_id=request.claim_id, status=BatchRequestStatus.ERRORED, response=None, error=error
                    )
                    continue

            result = await self.authenticated_user_query.get_user_by_id(user_id=request.user_id)
            match result:
                case AuthenticatedUserView() as user if not user.has_expired:
                    pass
                case _:
                    error = BatchRequestError(type="AccountExpiredError", message="The account that created the batch is expired.")
                    await self.batch_repository.complete_request(
                        claim_id=request.claim_id, status=BatchRequestStatus.ERRORED, response=None, error=error
                    )
                    continue

            claims.append(BatchRequestClaim(request=request, authenticated_user=user, authenticated_key=key))

        return ClaimBatchRequestsUseCaseSuccess(claims=claims)

    @staticmethod
    def _is_usable(key: Key) -> bool:
        return not key.revoked and (key.expires is None or key.expires > datetime.now(tz=UTC))
