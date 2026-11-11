from unittest.mock import AsyncMock

from httpx import AsyncClient
import pytest
import pytest_asyncio

from api.dependencies import cancel_batch_use_case_factory
from api.domain.batch.entities import BatchProcessingStatus, BatchRequestStatus
from api.domain.batch.errors import BatchAlreadyEndedError, BatchNotFoundError
from api.infrastructure.fastapi.routes import EndpointRoute
from api.tests.helpers import create_key
from api.tests.integration.factories.sql import BatchRequestSQLFactory, BatchSQLFactory, UserSQLFactory

URL = f"/v1{EndpointRoute.CHAT_COMPLETIONS_BATCHES}"


@pytest.mark.asyncio(loop_scope="session")
class TestCancelBatch:
    @pytest_asyncio.fixture(autouse=True)
    async def setup(self, db_session):
        self.user = UserSQLFactory(regular_user=True)
        self.key = await create_key(db_session, name="batch_key", user=self.user)

    async def test_happy_path(self, client: AsyncClient, db_session):
        batch = BatchSQLFactory(key=self.key)
        BatchRequestSQLFactory(batch=batch, status=BatchRequestStatus.PENDING)
        await db_session.flush()

        response = await client.post(url=f"{URL}/{batch.id}/cancel", headers={"Authorization": f"Bearer {self.key.token}"})

        assert response.status_code == 200, response.text
        data = response.json()
        assert data["processing_status"] == "canceling"
        assert data["request_counts"]["canceled"] == 1

    async def test_returns_not_found_for_another_users_batch(self, client: AsyncClient, db_session):
        other_batch = BatchSQLFactory(processing_status=BatchProcessingStatus.ENDED)
        await db_session.flush()

        response = await client.post(url=f"{URL}/{other_batch.id}/cancel", headers={"Authorization": f"Bearer {self.key.token}"})

        assert response.status_code == 404, response.text
        assert response.json().get("detail") == f"Batch {other_batch.id} not found."

    @pytest.mark.parametrize(
        "use_case_result,expected_status,expected_detail",
        [
            (BatchNotFoundError(id=42), 404, "Batch 42 not found."),
            (BatchAlreadyEndedError(id=42), 409, "Batch 42 has already ended."),
        ],
    )
    async def test_error_maps_to_correct_http_status(self, client: AsyncClient, app, use_case_result, expected_status, expected_detail):
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = use_case_result
        app.dependency_overrides[cancel_batch_use_case_factory] = lambda: mock_use_case

        response = await client.post(url=f"{URL}/42/cancel", headers={"Authorization": f"Bearer {self.key.token}"})

        assert response.status_code == expected_status
        assert response.json().get("detail") == expected_detail
