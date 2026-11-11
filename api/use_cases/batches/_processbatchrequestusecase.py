from dataclasses import dataclass
from enum import StrEnum
import math
import random
from typing import ClassVar

from api.domain.batch import BatchRepository
from api.domain.batch.entities import BatchRequestError, BatchRequestStatus, ClaimedBatchRequest
from api.domain.chat.entities import CreateChatCompletionsBody
from api.domain.model import ModelEnvironmentalImpactsComputer, ModelTokenizer
from api.domain.model.errors import StatusCodeModelError, TooBusyModelError, UnknownModelError
from api.domain.provider import (
    ProviderClient,
    ProviderConcurrencyLimiter,
    ProviderLimitsCannotBeEnforced,
    ProviderRepository,
    ProviderReservation,
    ProviderReservationRefused,
)
from api.domain.provider.entities import Provider, ProviderEndpoint, ProviderRequest, ProviderResponse
from api.domain.router import RouterRateLimiter, RouterRepository
from api.domain.router.entities import Router, RouterLoadBalancingStrategy, RouterType
from api.domain.router.errors import RouterHasNoProvidersError, RouterNotFoundError
from api.domain.usage import UsageContext, UsageRepository
from api.domain.user.errors import UserHasNoAccessToRouterError
from api.use_cases._providerrequestforwardingusecase import ForwardingCommand, ProviderRequestForwardingUseCase


class ProcessBatchRequestCommand(ForwardingCommand[CreateChatCompletionsBody]):
    batch_request: ClaimedBatchRequest


class BatchRequestRequeueReason(StrEnum):
    NO_CAPACITY = "no_capacity"
    RETRY = "retry"


@dataclass
class ProcessBatchRequestUseCaseSuccess:
    request_id: int


@dataclass
class ProcessBatchRequestUseCaseRequeued:
    request_id: int
    reason: BatchRequestRequeueReason


@dataclass
class ProcessBatchRequestUseCaseFailed:
    request_id: int
    error: BatchRequestError


type ProcessBatchRequestUseCaseResult = ProcessBatchRequestUseCaseSuccess | ProcessBatchRequestUseCaseRequeued | ProcessBatchRequestUseCaseFailed


class ProcessBatchRequestUseCase(ProviderRequestForwardingUseCase[ProcessBatchRequestCommand, ProcessBatchRequestUseCaseResult]):
    """Forward one batch request with the capacity interactive traffic leaves free.

    Its own execute(): unlike an interactive request, a batch request never waits for a provider nor answers 429. When
    no provider has spare capacity it goes back to the queue, and a transient provider failure is retried later.
    Batch requests are neither checked against nor charged to the interactive rate limits."""

    ROUTER_TYPE = RouterType.TEXT_GENERATION
    ENDPOINT = ProviderEndpoint.CHAT_COMPLETIONS
    MAX_ATTEMPTS: ClassVar[int] = 3
    RETRY_DELAY_SECONDS: ClassVar[float] = 30.0
    NO_CAPACITY_DELAY_SECONDS: ClassVar[float] = 1.0

    def __init__(
        self,
        model_environmental_impacts_computer: ModelEnvironmentalImpactsComputer,
        model_tokenizer: ModelTokenizer,
        provider_client: ProviderClient,
        provider_concurrency_limiter: ProviderConcurrencyLimiter,
        provider_repository: ProviderRepository,
        router_rate_limiter: RouterRateLimiter,
        router_repository: RouterRepository,
        usage_context: UsageContext,
        usage_repository: UsageRepository,
        random_generator: random.Random,
        batch_repository: BatchRepository,
        provider_capacity_share: float,
        provider_default_concurrency: int,
    ) -> None:
        super().__init__(
            model_environmental_impacts_computer=model_environmental_impacts_computer,
            model_tokenizer=model_tokenizer,
            provider_client=provider_client,
            provider_concurrency_limiter=provider_concurrency_limiter,
            provider_repository=provider_repository,
            router_rate_limiter=router_rate_limiter,
            router_repository=router_repository,
            usage_context=usage_context,
            usage_repository=usage_repository,
            random_generator=random_generator,
        )
        self.batch_repository = batch_repository
        self.provider_capacity_share = provider_capacity_share
        self.provider_default_concurrency = provider_default_concurrency

    async def execute(self, command: ProcessBatchRequestCommand) -> ProcessBatchRequestUseCaseResult:
        batch_request = command.batch_request

        result = await self._get_usable_router(authenticated_user=command.authenticated_user, model_name_or_alias=command.model)
        match result:
            case Router() as router:
                pass
            case error:
                return await self._complete_as_errored(batch_request=batch_request, error=error, model=command.model)

        estimated_prompt_tokens = self.model_tokenizer.compute_tokens(texts=command.get_prompts())
        request_id = self.usage_context.get_request_id()

        providers = await self.provider_repository.get_all_providers_of_router(router_id=router.id)
        result = await self._reserve_spare_capacity(providers=providers, strategy=router.load_balancing_strategy, request_id=request_id)
        match result:
            case ProviderReservation() as reservation:
                provider = reservation.provider
            case ProviderReservationRefused() | ProviderLimitsCannotBeEnforced():
                # when the load cannot be read either, waiting is safer than adding load to a provider that may be full
                await self.batch_repository.requeue_request(
                    request_id=batch_request.id, delay_seconds=self.NO_CAPACITY_DELAY_SECONDS, count_attempt=False
                )
                return ProcessBatchRequestUseCaseRequeued(request_id=batch_request.id, reason=BatchRequestRequeueReason.NO_CAPACITY)

        try:
            self.usage_context.record_provider(provider_id=provider.id, provider_model_name=provider.model_name)
            provider_request = ProviderRequest(id=request_id, endpoint=self.ENDPOINT, payload=command.payload)
            result = await self.provider_client.forward(provider=provider, request=provider_request)
            latency = self.usage_context.elapsed_ms()
        finally:
            await self.provider_concurrency_limiter.release(reservation=reservation)

        match result:
            case ProviderResponse() as provider_response:
                pass
            case TooBusyModelError() | UnknownModelError() | StatusCodeModelError() as error if self._is_transient(error) and (
                batch_request.attempts + 1 < self.MAX_ATTEMPTS
            ):
                delay_seconds = self.RETRY_DELAY_SECONDS * 2**batch_request.attempts
                await self.batch_repository.requeue_request(request_id=batch_request.id, delay_seconds=delay_seconds, count_attempt=True)
                return ProcessBatchRequestUseCaseRequeued(request_id=batch_request.id, reason=BatchRequestRequeueReason.RETRY)
            case error:
                return await self._complete_as_errored(batch_request=batch_request, error=error, model=command.model)

        billed_tokens = self._get_billed_tokens(
            estimated_prompt_tokens=estimated_prompt_tokens,
            completions=provider_response.get_completions(),
            provider_usage=provider_response.get_usage(),
        )
        usage = self._compute_usage(provider=provider, router=router, billed_tokens=billed_tokens, latency=latency)
        provider_response = provider_response.with_usage(usage)
        self._record_and_save_usage(usage=usage, latency=latency)

        await self.batch_repository.complete_request(
            request_id=batch_request.id, status=BatchRequestStatus.SUCCEEDED, response=provider_response.data.model_dump(), error=None
        )
        return ProcessBatchRequestUseCaseSuccess(request_id=batch_request.id)

    async def _complete_as_errored(self, batch_request: ClaimedBatchRequest, error: object, model: str) -> ProcessBatchRequestUseCaseFailed:
        batch_request_error = self._to_batch_request_error(error=error, model=model)
        await self.batch_repository.complete_request(
            request_id=batch_request.id, status=BatchRequestStatus.ERRORED, response=None, error=batch_request_error
        )
        return ProcessBatchRequestUseCaseFailed(request_id=batch_request.id, error=batch_request_error)

    async def _reserve_spare_capacity(
        self, providers: list[Provider], strategy: RouterLoadBalancingStrategy, request_id: str
    ) -> ProviderReservation | ProviderReservationRefused | ProviderLimitsCannotBeEnforced:
        open_providers = [provider for provider in providers if not provider.is_closed]
        if not open_providers:
            return ProviderReservationRefused()

        result = await self._order_by_strategy(providers=open_providers, strategy=strategy)
        match result:
            case list(ordered_providers):
                pass
            case limits_cannot_be_enforced:
                return limits_cannot_be_enforced

        for provider in ordered_providers:
            # the provider's load counts interactive and batch requests alike: below this limit, the rest stays free for interactive traffic
            result = await self.provider_concurrency_limiter.reserve(request_id=request_id, provider=provider, limit=self._batch_limit(provider))
            match result:
                case ProviderReservationRefused():
                    continue
                case reservation_or_limits_cannot_be_enforced:
                    return reservation_or_limits_cannot_be_enforced

        return ProviderReservationRefused()

    def _batch_limit(self, provider: Provider) -> int:
        if provider.qos_limit is None:
            return self.provider_default_concurrency

        return max(1, math.floor(provider.qos_limit * self.provider_capacity_share))

    @staticmethod
    def _is_transient(error: TooBusyModelError | UnknownModelError | StatusCodeModelError) -> bool:
        match error:
            case StatusCodeModelError(status_code=status_code):
                return status_code == 429 or status_code >= 500
            case _:
                return True

    @staticmethod
    def _to_batch_request_error(error: object, model: str) -> BatchRequestError:
        match error:
            # as on the interactive endpoints, a model the user cannot access is reported as not found
            case RouterNotFoundError() | RouterHasNoProvidersError() | UserHasNoAccessToRouterError():
                return BatchRequestError(type="ModelNotFoundError", message=f"Model {model} not found.", status_code=404)
            case StatusCodeModelError(status_code=status_code, detail=detail) | TooBusyModelError(status_code=status_code, detail=detail):
                return BatchRequestError(type=type(error).__name__, message=str(detail), status_code=status_code)
            case UnknownModelError(status_code=status_code, detail=detail):
                return BatchRequestError(type=type(error).__name__, message=str(detail), status_code=status_code)
            case _:
                return BatchRequestError(type=type(error).__name__, message=str(error))
