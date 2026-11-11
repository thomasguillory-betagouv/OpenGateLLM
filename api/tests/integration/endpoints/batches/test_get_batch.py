from unittest.mock import AsyncMock

from httpx import AsyncClient
import pytest
import pytest_asyncio

from api.dependencies import get_one_batch_use_case_factory
from api.domain.batch.entities import BatchRequestStatus
from api.domain.batch.errors import BatchNotFoundError
from api.infrastructure.fastapi.routes import EndpointRoute
from api.tests.helpers import create_key
from api.tests.integration.factories.sql import BatchRequestSQLFactory, BatchSQLFactory, UserSQLFactory

URL = f"/v1{EndpointRoute.CHAT_COMPLETIONS_BATCHES}"


@pytest.mark.asyncio(loop_scope="session")
class TestGetBatch:
    @pytest_asyncio.fixture(autouse=True)
    async def setup(self, db_session):
        self.user = UserSQLFactory(regular_user=True)
        self.key = await create_key(db_session, name="batch_key", user=self.user)

    async def test_happy_path(self, client: AsyncClient, db_session):
        batch = BatchSQLFactory(key=self.key)
        BatchRequestSQLFactory(batch=batch, status=BatchRequestStatus.SUCCEEDED)
        BatchRequestSQLFactory(batch=batch, status=BatchRequestStatus.PENDING)
        await db_session.flush()

        response = await client.get(url=f"{URL}/{batch.id}", headers={"Authorization": f"Bearer {self.key.token}"})

        assert response.status_code == 200, response.text
        data = response.json()
        assert data["object"] == "batch"
        assert data["id"] == batch.id
        assert data["request_counts"]["succeeded"] == 1
        assert data["request_counts"]["processing"] == 1

    async def test_error_maps_to_correct_http_status(self, client: AsyncClient, app):
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = BatchNotFoundError(id=42)
        app.dependency_overrides[get_one_batch_use_case_factory] = lambda: mock_use_case

        response = await client.get(url=f"{URL}/42", headers={"Authorization": f"Bearer {self.key.token}"})

        assert response.status_code == 404
        assert response.json().get("detail") == "Batch 42 not found."
