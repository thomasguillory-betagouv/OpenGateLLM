import asyncio
from collections.abc import Awaitable, Callable
import logging
import time
from typing import ClassVar

from api.use_cases.batches import (
    BatchRequestClaim,
    BatchRequestRequeueReason,
    ProcessBatchRequestUseCaseFailed,
    ProcessBatchRequestUseCaseRequeued,
    ProcessBatchRequestUseCaseSuccess,
    ReconcileBatchesUseCaseSuccess,
)

logger = logging.getLogger(__name__)

type ClaimBatchRequests = Callable[[int], Awaitable[list[BatchRequestClaim]]]
type ProcessBatchRequest = Callable[[BatchRequestClaim], Awaitable[ProcessBatchRequestUseCaseSuccess | ProcessBatchRequestUseCaseRequeued | ProcessBatchRequestUseCaseFailed]]  # fmt: off
type ReconcileBatches = Callable[[], Awaitable[ReconcileBatchesUseCaseSuccess]]


class BatchWorker:
    """Claim batch requests and process up to `concurrency` of them at a time, until `stop` is set.

    Several workers may run side by side: each claim locks its rows (SKIP LOCKED), so no request is processed twice
    unless a worker dies, in which case its claims go back to the queue once their lease expires."""

    POLL_INTERVAL_SECONDS: ClassVar[float] = 1.0
    NO_CAPACITY_PAUSE_SECONDS: ClassVar[float] = 0.5
    RECONCILE_INTERVAL_SECONDS: ClassVar[float] = 2.0
    SHUTDOWN_TIMEOUT_SECONDS: ClassVar[float] = 30.0

    def __init__(self, concurrency: int, claim: ClaimBatchRequests, process: ProcessBatchRequest, reconcile: ReconcileBatches) -> None:
        self.concurrency = concurrency
        self.claim = claim
        self.process = process
        self.reconcile = reconcile
        self._claims_paused_until = 0.0

    async def run(self, stop: asyncio.Event) -> None:
        reconciler = asyncio.create_task(self._reconcile_until(stop), name="batch-reconciler")
        in_flight: set[asyncio.Task] = set()
        try:
            while not stop.is_set():
                claims = await self._claim_free_slots(in_flight_count=len(in_flight))
                for claim in claims:
                    task = asyncio.create_task(self._process(claim), name=f"batch-request-{claim.request.id}")
                    in_flight.add(task)
                    task.add_done_callback(in_flight.discard)

                if not claims:
                    await self._wait_for_a_change(in_flight=in_flight, stop=stop)
        finally:
            reconciler.cancel()
            await self._drain(in_flight)

    async def _claim_free_slots(self, in_flight_count: int) -> list[BatchRequestClaim]:
        free_slots = self.concurrency - in_flight_count
        if free_slots <= 0 or time.monotonic() < self._claims_paused_until:
            return []

        try:
            return await self.claim(free_slots)
        except Exception:
            logger.exception("Failed to claim batch requests.")
            return []

    async def _process(self, claim: BatchRequestClaim) -> None:
        try:
            result = await self.process(claim)
        except Exception:
            # the request stays claimed: it goes back to the queue once its lease expires
            logger.exception("Failed to process batch request %s.", claim.request.id)
            return

        match result:
            case ProcessBatchRequestUseCaseRequeued(reason=BatchRequestRequeueReason.NO_CAPACITY):
                # the providers are busy with interactive traffic: claiming more now would only requeue them too
                self._claims_paused_until = time.monotonic() + self.NO_CAPACITY_PAUSE_SECONDS
            case ProcessBatchRequestUseCaseFailed(request_id=request_id, error=error):
                logger.info("Batch request %s failed: %s.", request_id, error.type)
            case _:
                pass

    async def _wait_for_a_change(self, in_flight: set[asyncio.Task], stop: asyncio.Event) -> None:
        stopped = asyncio.create_task(stop.wait())
        timeout = max(self._claims_paused_until - time.monotonic(), 0.0) or self.POLL_INTERVAL_SECONDS
        try:
            await asyncio.wait({stopped, *in_flight}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        finally:
            stopped.cancel()

    async def _reconcile_until(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                result = await self.reconcile()
                if result.released_requests or result.expired_requests or result.ended_batches:
                    logger.info(
                        "Batches reconciled: %s claims released, %s requests expired, %s batches ended.",
                        result.released_requests,
                        result.expired_requests,
                        result.ended_batches,
                    )
            except Exception:
                logger.exception("Failed to reconcile batches.")

            try:
                await asyncio.wait_for(stop.wait(), timeout=self.RECONCILE_INTERVAL_SECONDS)
            except TimeoutError:
                pass

    async def _drain(self, in_flight: set[asyncio.Task]) -> None:
        if not in_flight:
            return

        logger.info("Waiting for %s batch requests in progress.", len(in_flight))
        _, unfinished = await asyncio.wait(set(in_flight), timeout=self.SHUTDOWN_TIMEOUT_SECONDS)
        for task in unfinished:
            task.cancel()
        if unfinished:
            logger.warning("%s batch requests were interrupted: they go back to the queue once their lease expires.", len(unfinished))
