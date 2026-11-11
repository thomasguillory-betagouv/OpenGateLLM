"""Batch worker: processes the requests of the batches with the capacity interactive traffic leaves free.

Composition root of the worker process, as lifespan.py is the API's. It runs from the API image, next to the API:

    python -m api.worker
"""

import asyncio
import logging
import random
import signal
from uuid import uuid4

from fastapi import BackgroundTasks
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import async_sessionmaker

from api.infrastructure.configuration import Configuration, get_configuration
from api.infrastructure.ecologit import EcologitModelEnvironmentalImpactsComputer
from api.infrastructure.fastapi import NoOpRequestTracer, UsageRecorder
from api.infrastructure.fastapi.routes import EndpointRoute
from api.infrastructure.http import HttpProviderAdapterBuilder, HttpProviderClient
from api.infrastructure.jwt import JwtKeyEncoder
from api.infrastructure.logging import configure_logging
from api.infrastructure.postgres import (
    PostgresAuthenticatedUserQuery,
    PostgresBatchRepository,
    PostgresKeyRepository,
    PostgresProviderRepository,
    PostgresRouterRepository,
    PostgresUsageRepository,
)
from api.infrastructure.redis import RedisProviderConcurrencyLimiter, RedisRouterRateLimiter
from api.infrastructure.tiktoken import TiktokenModelTokenizer
from api.infrastructure.worker import BatchWorker
from api.lifespan import (
    USAGE_SAVES_SHUTDOWN_TIMEOUT_SECONDS,
    create_autocommit_postgres_session_factory,
    create_postgres_engine,
    create_redis_pool,
    initialize_tokenizer,
)
from api.use_cases.batches import (
    BatchRequestClaim,
    ClaimBatchRequestsCommand,
    ClaimBatchRequestsUseCase,
    ExtendBatchRequestClaimsCommand,
    ExtendBatchRequestClaimsUseCase,
    ProcessBatchRequestCommand,
    ProcessBatchRequestUseCase,
    ProcessBatchRequestUseCaseFailed,
    ProcessBatchRequestUseCaseRequeued,
    ProcessBatchRequestUseCaseSuccess,
    ReconcileBatchesUseCase,
    ReconcileBatchesUseCaseSuccess,
)

logger = logging.getLogger(__name__)

# the worker extends the claims it holds every third of the lease: a claim expires about a minute after its worker dies
CLAIM_LEASE_SECONDS = 60
# usage.endpoint holds the route the consumption came through: GET /v1/usage tells batch usage apart this way
BATCH_USAGE_ENDPOINT = f"/v1{EndpointRoute.CHAT_COMPLETIONS_BATCHES}"


async def run(configuration: Configuration, stop: asyncio.Event) -> None:
    settings = configuration.settings
    redis_pool = await create_redis_pool(configuration)
    postgres_engine = create_postgres_engine(configuration)
    session_factory: async_sessionmaker = create_autocommit_postgres_session_factory(engine=postgres_engine)

    redis_client = Redis(connection_pool=redis_pool)
    key_encoder = JwtKeyEncoder(secret_key=settings.auth_secret_key)
    model_tokenizer = TiktokenModelTokenizer(model=initialize_tokenizer(configuration=configuration))
    model_environmental_impacts_computer = EcologitModelEnvironmentalImpactsComputer()
    provider_client = HttpProviderClient(adapter_builder=HttpProviderAdapterBuilder())
    provider_concurrency_limiter = RedisProviderConcurrencyLimiter(redis_client=redis_client)
    # never called: batch requests are neither checked against nor charged to the interactive rate limits
    router_rate_limiter = RedisRouterRateLimiter(
        background_tasks=BackgroundTasks(),
        redis_pool=redis_pool,
        strategy=settings.rate_limiting_strategy,
        cached_tokens_discount=settings.rate_limiting_cached_tokens_discount,
    )

    async def claim(limit: int) -> list[BatchRequestClaim]:
        async with session_factory() as session:
            use_case = ClaimBatchRequestsUseCase(
                batch_repository=PostgresBatchRepository(postgres_session=session),
                key_repository=PostgresKeyRepository(key_encoder=key_encoder, postgres_session=session),
                authenticated_user_query=PostgresAuthenticatedUserQuery(postgres_session=session),
            )
            result = await use_case.execute(ClaimBatchRequestsCommand(limit=limit, lease_seconds=CLAIM_LEASE_SECONDS))
            return result.claims

    async def process(
        claim: BatchRequestClaim,
    ) -> ProcessBatchRequestUseCaseSuccess | ProcessBatchRequestUseCaseRequeued | ProcessBatchRequestUseCaseFailed:
        async with session_factory() as session:
            usage_recorder = UsageRecorder.open(
                tracer=NoOpRequestTracer(),
                request_id=uuid4().hex,
                endpoint=BATCH_USAGE_ENDPOINT,
                user=claim.authenticated_user,
                key=claim.authenticated_key,
            )
            use_case = ProcessBatchRequestUseCase(
                model_environmental_impacts_computer=model_environmental_impacts_computer,
                model_tokenizer=model_tokenizer,
                provider_client=provider_client,
                provider_concurrency_limiter=provider_concurrency_limiter,
                provider_repository=PostgresProviderRepository(postgres_session=session),
                router_rate_limiter=router_rate_limiter,
                router_repository=PostgresRouterRepository(postgres_session=session),
                usage_context=usage_recorder,
                usage_repository=PostgresUsageRepository(postgres_session=session, session_factory=session_factory),
                random_generator=random.Random(),
                batch_repository=PostgresBatchRepository(postgres_session=session),
                provider_capacity_share=settings.batch_provider_capacity_share,
                provider_default_concurrency=settings.batch_provider_default_concurrency,
            )
            command = ProcessBatchRequestCommand(
                payload=claim.request.params,
                authenticated_user=claim.authenticated_user,
                authenticated_key=claim.authenticated_key,
                batch_request=claim.request,
            )
            result = await use_case.execute(command)
            usage_recorder.close(status_code=200 if isinstance(result, ProcessBatchRequestUseCaseSuccess) else None)
            return result

    async def reconcile() -> ReconcileBatchesUseCaseSuccess:
        async with session_factory() as session:
            return await ReconcileBatchesUseCase(batch_repository=PostgresBatchRepository(postgres_session=session)).execute()

    async def extend_claims(claim_ids: list[str]) -> None:
        async with session_factory() as session:
            use_case = ExtendBatchRequestClaimsUseCase(batch_repository=PostgresBatchRepository(postgres_session=session))
            await use_case.execute(ExtendBatchRequestClaimsCommand(claim_ids=claim_ids, lease_seconds=CLAIM_LEASE_SECONDS))

    worker = BatchWorker(
        concurrency=settings.batch_worker_concurrency,
        heartbeat_interval_seconds=CLAIM_LEASE_SECONDS / 3,
        claim=claim,
        process=process,
        reconcile=reconcile,
        extend_claims=extend_claims,
    )
    logger.info("Batch worker started (concurrency: %s).", settings.batch_worker_concurrency)
    try:
        await worker.run(stop=stop)
    finally:
        await PostgresUsageRepository.wait_for_pending_saves(timeout=USAGE_SAVES_SHUTDOWN_TIMEOUT_SECONDS)
        await redis_client.aclose()
        await redis_pool.aclose()
        await postgres_engine.dispose()
        logger.info("Batch worker stopped.")


async def main() -> None:
    configuration = get_configuration()
    configure_logging(configuration.settings)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for stop_signal in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(stop_signal, stop.set)

    await run(configuration=configuration, stop=stop)


if __name__ == "__main__":
    asyncio.run(main())
