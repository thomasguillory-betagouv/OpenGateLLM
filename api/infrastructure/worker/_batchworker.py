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
type ExtendClaims = Callable[[list[str]], Awaitable[None]]


class BatchWorker:
    """Claim batch requests and process up to `concurrency` of them at a time, until `stop` is set.

    Several workers may run side by side: each claim locks its rows (SKIP LOCKED), so no request is processed twice
    unless a worker dies. Its claims then go back to the queue once their lease expires: a live worker extends the
    leases of the requests it processes every `heartbeat_interval_seconds`, a dead one stops."""

    POLL_INTERVAL_SECONDS: ClassVar[float] = 1.0
    NO_CAPACITY_PAUSE_SECONDS: ClassVar[float] = 0.5
    RECONCILE_INTERVAL_SECONDS: ClassVar[float] = 2.0
    SHUTDOWN_TIMEOUT_SECONDS: ClassVar[float] = 30.0

    def __init__(
        self,
        concurrency: int,
        heartbeat_interval_seconds: float,
        claim: ClaimBatchRequests,
        process: ProcessBatchRequest,
        reconcile: ReconcileBatches,
        extend_claims: ExtendClaims,
    ) -> None:
        self.concurrency = concurrency
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self.claim = claim
        self.process = process
        self.reconcile = reconcile
        self.extend_claims = extend_claims
        self._claims_paused_until = 0.0
        self._in_flight: dict[asyncio.Task, str] = {}

    async def run(self, stop: asyncio.Event) -> None:
        reconciler = asyncio.create_task(self._reconcile_until(stop), name="batch-reconciler")
        heartbeat = asyncio.create_task(self._extend_claims_forever(), name="batch-heartbeat")
        try:
            while not stop.is_set():
                claims = await self._claim_free_slots(in_flight_count=len(self._in_flight))
                for claim in claims:
                    task = asyncio.create_task(self._process(claim), name=f"batch-request-{claim.request.id}")
                    self._in_flight[task] = claim.request.claim_id
                    task.add_done_callback(self._in_flight.pop)

                if not claims:
                    await self._wait_for_a_change(stop=stop)
        finally:
            await self._cancel_and_wait(reconciler)
            # the heartbeat keeps the claims of the requests still running until they finish or are interrupted
            await self._drain()
            await self._cancel_and_wait(heartbeat)

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
            # the request goes back to the queue once its claim expires, which counts an attempt
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

    async def _wait_for_a_change(self, stop: asyncio.Event) -> None:
        stopped = asyncio.create_task(stop.wait())
        timeout = max(self._claims_paused_until - time.monotonic(), 0.0) or self.POLL_INTERVAL_SECONDS
        try:
            await asyncio.wait({stopped, *self._in_flight}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        finally:
            stopped.cancel()

    async def _extend_claims_forever(self) -> None:
        while True:
            await asyncio.sleep(self.heartbeat_interval_seconds)
            try:
                await self.extend_claims(list(self._in_flight.values()))
            except Exception:
                logger.exception("Failed to extend the claims of the batch requests in progress.")

    async def _reconcile_until(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                result = await self.reconcile()
                if result.released_requests or result.canceled_requests or result.expired_requests or result.ended_batches:
                    logger.info(
                        "Batches reconciled: %s claims released, %s requests canceled, %s requests expired, %s batches ended.",
                        result.released_requests,
                        result.canceled_requests,
                        result.expired_requests,
                        result.ended_batches,
                    )
            except Exception:
                logger.exception("Failed to reconcile batches.")

            try:
                await asyncio.wait_for(stop.wait(), timeout=self.RECONCILE_INTERVAL_SECONDS)
            except TimeoutError:
                pass

    async def _drain(self) -> None:
        if not self._in_flight:
            return

        logger.info("Waiting for %s batch requests in progress.", len(self._in_flight))
        _, unfinished = await asyncio.wait(set(self._in_flight), timeout=self.SHUTDOWN_TIMEOUT_SECONDS)
        if not unfinished:
            return

        for task in unfinished:
            task.cancel()
        # their finally blocks (provider reservation release) must run before the caller closes Redis and Postgres
        await asyncio.gather(*unfinished, return_exceptions=True)
        logger.warning("%s batch requests were interrupted: they go back to the queue once their claim expires.", len(unfinished))

    @staticmethod
    async def _cancel_and_wait(task: asyncio.Task) -> None:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
