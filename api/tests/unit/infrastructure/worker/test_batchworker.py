import asyncio
from unittest.mock import AsyncMock

import pytest

from api.domain.batch.entities import BatchRequestStatus, ClaimedBatchRequest
from api.infrastructure.worker import BatchWorker
from api.tests.unit.use_case.factories import AuthenticatedUserFactory, KeyFactory
from api.use_cases.batches import (
    BatchRequestClaim,
    BatchRequestRequeueReason,
    ProcessBatchRequestUseCaseRequeued,
    ProcessBatchRequestUseCaseSuccess,
    ReconcileBatchesUseCaseSuccess,
)


def _claim(request_id: int) -> BatchRequestClaim:
    request = ClaimedBatchRequest(
        id=request_id,
        batch_id=1,
        custom_id=f"request-{request_id}",
        position=request_id,
        params={"model": "m"},
        status=BatchRequestStatus.PROCESSING,
        attempts=0,
        claim_id=f"claim-{request_id}",
        user_id=1,
        key_id=2,
    )
    return BatchRequestClaim(request=request, authenticated_user=AuthenticatedUserFactory(id=1), authenticated_key=KeyFactory(id=2))


def _worker(mock_claim: AsyncMock, mock_process: AsyncMock, mock_extend_claims: AsyncMock | None = None, concurrency: int = 2) -> BatchWorker:
    mock_reconcile = AsyncMock(
        return_value=ReconcileBatchesUseCaseSuccess(released_requests=0, canceled_requests=0, expired_requests=0, ended_batches=0)
    )
    worker = BatchWorker(
        concurrency=concurrency,
        heartbeat_interval_seconds=0.01,
        claim=mock_claim,
        process=mock_process,
        reconcile=mock_reconcile,
        extend_claims=mock_extend_claims or AsyncMock(),
    )
    worker.POLL_INTERVAL_SECONDS = 0.01
    return worker


async def _run_until(worker: BatchWorker, condition, before_stopping=None) -> None:
    stop = asyncio.Event()
    run = asyncio.create_task(worker.run(stop))
    for _ in range(200):
        if condition():
            break
        await asyncio.sleep(0.01)
    if before_stopping is not None:
        before_stopping()
    stop.set()
    await asyncio.wait_for(run, timeout=5)


class TestBatchWorker:
    @pytest.mark.asyncio
    async def test_should_process_the_claimed_requests_within_its_concurrency(self):
        # Arrange
        claims = [_claim(1), _claim(2)]
        mock_claim = AsyncMock(side_effect=lambda limit: [claims.pop(0) for _ in range(min(limit, len(claims)))])
        mock_process = AsyncMock(side_effect=lambda claim: ProcessBatchRequestUseCaseSuccess(request_id=claim.request.id))
        worker = _worker(mock_claim=mock_claim, mock_process=mock_process)

        # Act
        await _run_until(worker, condition=lambda: mock_process.await_count == 2)

        # Assert
        assert [call.args[0].request.id for call in mock_process.await_args_list] == [1, 2]
        assert mock_claim.await_args_list[0].args == (2,)

    @pytest.mark.asyncio
    async def test_should_pause_its_claims_when_providers_have_no_spare_capacity(self):
        # Arrange
        mock_claim = AsyncMock(side_effect=lambda limit: [_claim(1)])
        mock_process = AsyncMock(return_value=ProcessBatchRequestUseCaseRequeued(request_id=1, reason=BatchRequestRequeueReason.NO_CAPACITY))
        worker = _worker(mock_claim=mock_claim, mock_process=mock_process, concurrency=1)
        worker.NO_CAPACITY_PAUSE_SECONDS = 60

        # Act
        await _run_until(worker, condition=lambda: mock_process.await_count >= 1)
        await asyncio.sleep(0.05)

        # Assert
        assert mock_claim.await_count == 1

    @pytest.mark.asyncio
    async def test_should_extend_the_claims_of_the_requests_in_progress(self):
        # Arrange
        release_the_request = asyncio.Event()

        async def process(claim: BatchRequestClaim) -> ProcessBatchRequestUseCaseSuccess:
            await release_the_request.wait()
            return ProcessBatchRequestUseCaseSuccess(request_id=claim.request.id)

        claims = [_claim(1)]
        mock_claim = AsyncMock(side_effect=lambda limit: [claims.pop()] if claims else [])
        mock_extend_claims = AsyncMock()
        worker = _worker(mock_claim=mock_claim, mock_process=AsyncMock(side_effect=process), mock_extend_claims=mock_extend_claims)

        # Act
        await _run_until(
            worker,
            condition=lambda: any(call.args == (["claim-1"],) for call in mock_extend_claims.await_args_list),
            before_stopping=release_the_request.set,
        )

        # Assert
        assert any(call.args == (["claim-1"],) for call in mock_extend_claims.await_args_list)
