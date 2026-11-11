import json
from unittest.mock import AsyncMock

from httpx import AsyncClient
import pytest
import pytest_asyncio

from api.dependencies import get_batch_results_use_case_factory
from api.domain.batch.entities import BatchProcessingStatus, BatchRequestStatus
from api.domain.batch.errors import BatchNotEndedError, BatchNotFoundError
from api.infrastructure.fastapi.routes import EndpointRoute
from api.tests.helpers import create_key
from api.tests.integration.factories.sql import BatchRequestSQLFactory, BatchSQLFactory, UserSQLFactory

URL = f"/v1{EndpointRoute.CHAT_COMPLETIONS_BATCHES}"


@pytest.mark.asyncio(loop_scope="session")
class TestGetBatchResults:
    @pytest_asyncio.fixture(autouse=True)
    async def setup(self, db_session):
        self.user = UserSQLFactory(regular_user=True)
        self.key = await create_key(db_session, name="batch_key", user=self.user)

    async def test_happy_path(self, client: AsyncClient, db_session):
        batch = BatchSQLFactory(key=self.key, processing_status=BatchProcessingStatus.ENDED)
        BatchRequestSQLFactory(
            batch=batch, custom_id="ok", position=0, status=BatchRequestStatus.SUCCEEDED, response={"id": "chatcmpl-1", "choices": []}
        )
        BatchRequestSQLFactory(
            batch=batch,
            custom_id="ko",
            position=1,
            status=BatchRequestStatus.ERRORED,
            error={"type": "StatusCodeModelError", "message": "Bad request", "status_code": 400},
        )
        await db_session.flush()

        response = await client.get(url=f"{URL}/{batch.id}/results", headers={"Authorization": f"Bearer {self.key.token}"})

        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "application/x-ndjson"
        lines = [json.loads(line) for line in response.text.splitlines()]
        assert lines == [
            {"custom_id": "ok", "result": {"type": "succeeded", "response": {"id": "chatcmpl-1", "choices": []}, "error": None}},
            {
                "custom_id": "ko",
                "result": {
                    "type": "errored",
                    "response": None,
                    "error": {"type": "StatusCodeModelError", "message": "Bad request", "status_code": 400},
                },
            },
        ]

    @pytest.mark.parametrize(
        "use_case_result,expected_status,expected_detail",
        [
            (BatchNotFoundError(id=42), 404, "Batch 42 not found."),
            (BatchNotEndedError(id=42), 409, "Batch 42 has not ended yet: its results are available once its processing_status is ended."),
        ],
    )
    async def test_error_maps_to_correct_http_status(self, client: AsyncClient, app, use_case_result, expected_status, expected_detail):
        mock_use_case = AsyncMock()
        mock_use_case.execute.return_value = use_case_result
        app.dependency_overrides[get_batch_results_use_case_factory] = lambda: mock_use_case

        response = await client.get(url=f"{URL}/42/results", headers={"Authorization": f"Bearer {self.key.token}"})

        assert response.status_code == expected_status
        assert response.json().get("detail") == expected_detail
