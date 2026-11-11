from unittest.mock import AsyncMock

from httpx import AsyncClient
import pytest
import pytest_asyncio

from api.dependencies import create_batch_use_case_factory
from api.domain.batch.errors import BatchCustomIdNotUniqueError
from api.domain.router.entities import RouterType
from api.domain.router.errors import RouterHasWrongTypeError, RouterNotFoundError
from api.infrastructure.fastapi.routes import EndpointRoute
from api.tests.helpers import INVALID_API_KEY, create_key
from api.tests.integration.factories.sql import RouterSQLFactory, UserSQLFactory

URL = f"/v1{EndpointRoute.CHAT_COMPLETIONS_BATCHES}"
MODEL_NAME = "batch-router"


def _batch_body(*custom_ids: str, **params_overrides) -> dict:
    params = {"model": MODEL_NAME, "messages": [{"role": "user", "content": "Hello"}], **params_overrides}
    return {"requests": [{"custom_id": custom_id, "params": params} for custom_id in custom_ids]}


@pytest.mark.asyncio(loop_scope="session")
class TestCreateBatch:
    @pytest_asyncio.fixture(autouse=True)
    async def setup(self, db_session):
        self.user = UserSQLFactory(admin_user=True)
        self.key = await create_key(db_session, name="batch_key", user=self.user)
        RouterSQLFactory(user=self.user, name=MODEL_NAME, type=RouterType.TEXT_GENERATION)
        await db_session.flush()

    async def test_happy_path(self, client: AsyncClient):
        response = await client.post(url=URL, json=_batch_body("request-1", "request-2"), headers={"Authorization": f"Bearer {self.key.token}"})

        assert response.status_code == 201, response.text
        data = response.json()
        assert data["object"] == "batch"
        assert data["processing_status"] == "in_progress"
        assert data["request_counts"] == {"processing": 2, "succeeded": 0, "errored": 0, "canceled": 0, "expired": 0}
        assert data["expires"] > data["created"]

    async def test_rejects_a_streamed_request(self, client: AsyncClient):
        response = await client.post(url=URL, json=_batch_body("request-1", stream=True), headers={"Authorization": f"Bearer {self.key.token}"})

        assert response.status_code == 422, response.text

    @pytest.mark.parametrize(
        "use_case_result,expected_status,expected_detail",
        [
            (BatchCustomIdNotUniqueError(custom_ids=["a", "b"]), 400, "custom_id must be unique within a batch. Duplicated: a, b."),
            (RouterNotFoundError(name="unknown-model"), 404, "Model unknown-model not found."),
            (
                RouterHasWrongTypeError(id=1, actual_type=RouterType.TEXT_EMBEDDINGS_INFERENCE, expected_type=RouterType.TEXT_GENERATION),
                422,
                f"Model has wrong type. Expected: {RouterType.TEXT_GENERATION}. Actual: {RouterType.TEXT_EMBEDDINGS_INFERENCE}.",
            ),
        ],
    )
    async def test_error_maps_to_correct_http_status(self, client: AsyncClient, app, use_case_result, expected_status, expected_detail):
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = use_case_result
        app.dependency_overrides[create_batch_use_case_factory] = lambda: mock_use_case

        response = await client.post(url=URL, json=_batch_body("request-1"), headers={"Authorization": f"Bearer {self.key.token}"})

        assert response.status_code == expected_status
        assert response.json().get("detail") == expected_detail

    @pytest.mark.parametrize(
        "headers,expected_status",
        [({}, 401), ({"Authorization": f"Bearer {INVALID_API_KEY}"}, 401)],
    )
    async def test_rejects_unauthenticated_requests(self, client: AsyncClient, headers, expected_status):
        response = await client.post(url=URL, json=_batch_body("request-1"), headers=headers)

        assert response.status_code == expected_status
