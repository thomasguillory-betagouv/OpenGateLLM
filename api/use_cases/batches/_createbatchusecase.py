from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import ClassVar

from api.domain.batch import BatchRepository
from api.domain.batch.entities import Batch, BatchRequestToCreate
from api.domain.batch.errors import BatchCustomIdNotUniqueError
from api.domain.key.entities import Key
from api.domain.router import RouterRepository
from api.domain.router.entities import Router, RouterType
from api.domain.router.errors import RouterHasWrongTypeError, RouterNotFoundError
from api.domain.user.views import AuthenticatedUserView


@dataclass
class CreateBatchCommand:
    authenticated_user: AuthenticatedUserView
    authenticated_key: Key
    requests: list[BatchRequestToCreate]


@dataclass
class CreateBatchUseCaseSuccess:
    batch: Batch


type CreateBatchUseCaseResult = CreateBatchUseCaseSuccess | BatchCustomIdNotUniqueError | RouterNotFoundError | RouterHasWrongTypeError


class CreateBatchUseCase:
    ROUTER_TYPE: ClassVar[RouterType] = RouterType.TEXT_GENERATION
    EXPIRATION: ClassVar[timedelta] = timedelta(hours=24)

    def __init__(self, batch_repository: BatchRepository, router_repository: RouterRepository):
        self.batch_repository = batch_repository
        self.router_repository = router_repository

    async def execute(self, command: CreateBatchCommand) -> CreateBatchUseCaseResult:
        authenticated_user = command.authenticated_user

        custom_id_counts = Counter(request.custom_id for request in command.requests)
        duplicated_custom_ids = sorted(custom_id for custom_id, count in custom_id_counts.items() if count > 1)
        if duplicated_custom_ids:
            return BatchCustomIdNotUniqueError(custom_ids=duplicated_custom_ids)

        # the model of each request is checked again when the request is processed: access may change in between
        for model_name_or_alias in sorted({request.params["model"] for request in command.requests}):
            result = await self.router_repository.get_router_by_name_or_alias(name_or_alias=model_name_or_alias)
            match result:
                case Router() as router:
                    pass
                case error:
                    return error

            if router.type != self.ROUTER_TYPE:
                return RouterHasWrongTypeError(id=router.id, actual_type=router.type, expected_type=self.ROUTER_TYPE)
            if authenticated_user.cannot_access_router(router_id=router.id):
                # as the interactive endpoints do, a model the user cannot access is reported as not found
                return RouterNotFoundError(name=model_name_or_alias)

        batch = await self.batch_repository.create_batch(
            user_id=authenticated_user.id,
            key_id=command.authenticated_key.id,
            expires=datetime.now(tz=UTC) + self.EXPIRATION,
            requests=command.requests,
        )

        return CreateBatchUseCaseSuccess(batch=batch)
