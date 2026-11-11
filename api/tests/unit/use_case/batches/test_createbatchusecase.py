from datetime import UTC, datetime
from unittest.mock import create_autospec

import pytest

from api.domain.batch import BatchRepository
from api.domain.batch.entities import BatchRequestToCreate
from api.domain.batch.errors import BatchCustomIdNotUniqueError
from api.domain.role.entities import Limit, LimitType
from api.domain.router import RouterRepository
from api.domain.router.entities import RouterType
from api.domain.router.errors import RouterHasWrongTypeError, RouterNotFoundError
from api.tests.unit.use_case.factories import AuthenticatedUserFactory, KeyFactory, RouterFactory
from api.use_cases.batches import CreateBatchCommand, CreateBatchUseCase, CreateBatchUseCaseSuccess


def _requests(*custom_ids: str, model: str = "chat-router") -> list[BatchRequestToCreate]:
    return [BatchRequestToCreate(custom_id=custom_id, params={"model": model, "messages": []}) for custom_id in custom_ids]


@pytest.fixture
def use_case():
    return CreateBatchUseCase(
        batch_repository=create_autospec(BatchRepository, instance=True, spec_set=True),
        router_repository=create_autospec(RouterRepository, instance=True, spec_set=True),
    )


@pytest.fixture
def user():
    return AuthenticatedUserFactory(id=1, without_permission=True, limits=[Limit(router_id=5, type=LimitType.RPM, value=10)])


class TestCreateBatchUseCase:
    @pytest.mark.asyncio
    async def test_should_create_the_batch_expiring_in_24_hours(self, use_case, user):
        # Arrange
        use_case.router_repository.get_router_by_name_or_alias.return_value = RouterFactory(id=5, type=RouterType.TEXT_GENERATION)
        requests = _requests("a", "b")
        key = KeyFactory(id=2)

        # Act
        before = datetime.now(tz=UTC)
        result = await use_case.execute(CreateBatchCommand(authenticated_user=user, authenticated_key=key, requests=requests))

        # Assert
        assert result == CreateBatchUseCaseSuccess(batch=use_case.batch_repository.create_batch.return_value)
        arguments = use_case.batch_repository.create_batch.await_args.kwargs
        assert arguments["user_id"] == 1
        assert arguments["key_id"] == 2
        assert arguments["requests"] == requests
        assert arguments["expires"] - before >= CreateBatchUseCase.EXPIRATION
        use_case.router_repository.get_router_by_name_or_alias.assert_awaited_once_with(name_or_alias="chat-router")

    @pytest.mark.asyncio
    async def test_should_reject_duplicated_custom_ids(self, use_case, user):
        # Act
        result = await use_case.execute(
            CreateBatchCommand(authenticated_user=user, authenticated_key=KeyFactory(), requests=_requests("a", "b", "a"))
        )

        # Assert
        assert result == BatchCustomIdNotUniqueError(custom_ids=["a"])
        use_case.batch_repository.create_batch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_should_return_the_error_of_an_unknown_model(self, use_case, user):
        # Arrange
        use_case.router_repository.get_router_by_name_or_alias.return_value = RouterNotFoundError(name="chat-router")

        # Act
        result = await use_case.execute(CreateBatchCommand(authenticated_user=user, authenticated_key=KeyFactory(), requests=_requests("a")))

        # Assert
        assert result == RouterNotFoundError(name="chat-router")
        use_case.batch_repository.create_batch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_should_reject_a_model_of_another_type(self, use_case, user):
        # Arrange
        use_case.router_repository.get_router_by_name_or_alias.return_value = RouterFactory(id=5, type=RouterType.TEXT_EMBEDDINGS_INFERENCE)

        # Act
        result = await use_case.execute(CreateBatchCommand(authenticated_user=user, authenticated_key=KeyFactory(), requests=_requests("a")))

        # Assert
        assert result == RouterHasWrongTypeError(id=5, actual_type=RouterType.TEXT_EMBEDDINGS_INFERENCE, expected_type=RouterType.TEXT_GENERATION)

    @pytest.mark.asyncio
    async def test_should_report_a_model_the_user_cannot_access_as_not_found(self, use_case, user):
        # Arrange
        use_case.router_repository.get_router_by_name_or_alias.return_value = RouterFactory(id=6, type=RouterType.TEXT_GENERATION)

        # Act
        result = await use_case.execute(CreateBatchCommand(authenticated_user=user, authenticated_key=KeyFactory(), requests=_requests("a")))

        # Assert
        assert result == RouterNotFoundError(name="chat-router")
        use_case.batch_repository.create_batch.assert_not_awaited()
