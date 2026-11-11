from httpx import AsyncClient
import pytest
import pytest_asyncio

from api.infrastructure.fastapi.routes import EndpointRoute
from api.tests.helpers import create_key
from api.tests.integration.factories.sql import BatchSQLFactory, UserSQLFactory

URL = f"/v1{EndpointRoute.CHAT_COMPLETIONS_BATCHES}"


@pytest.mark.asyncio(loop_scope="session")
class TestGetBatches:
    @pytest_asyncio.fixture(autouse=True)
    async def setup(self, db_session):
        self.user = UserSQLFactory(regular_user=True)
        self.key = await create_key(db_session, name="batch_key", user=self.user)

    async def test_happy_path(self, client: AsyncClient, db_session):
        batch = BatchSQLFactory(key=self.key)
        BatchSQLFactory()
        await db_session.flush()

        response = await client.get(url=URL, headers={"Authorization": f"Bearer {self.key.token}"})

        assert response.status_code == 200, response.text
        data = response.json()
        assert data["object"] == "list"
        assert data["total"] == 1
        assert [item["id"] for item in data["data"]] == [batch.id]
