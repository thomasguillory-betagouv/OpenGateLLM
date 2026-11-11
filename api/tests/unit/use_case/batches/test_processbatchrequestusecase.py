import random
from unittest.mock import create_autospec
from uuid import uuid4

import pytest

from api.domain.batch import BatchRepository
from api.domain.batch.entities import BatchRequestError, BatchRequestStatus, ClaimedBatchRequest
from api.domain.chat.entities import ChatCompletion
from api.domain.model import ModelEnvironmentalImpactsComputer, ModelTokenizer
from api.domain.model.errors import StatusCodeModelError, TooBusyModelError
from api.domain.provider import (
    ProviderClient,
    ProviderConcurrencyLimiter,
    ProviderLimitsCannotBeEnforced,
    ProviderRepository,
    ProviderReservation,
    ProviderReservationRefused,
)
from api.domain.provider.entities import ProviderEndpoint, ProviderRequest, ProviderResponse, ProviderType
from api.domain.router import RouterRateLimiter, RouterRepository
from api.domain.router.entities import RouterLoadBalancingStrategy, RouterType
from api.domain.router.errors import RouterNotFoundError
from api.domain.usage import UsageContext, UsageRepository
from api.domain.usage.entities import EnvironmentalImpacts, Usage
from api.tests.unit.use_case.factories import AuthenticatedUserFactory, KeyFactory, ProviderFactory, RouterFactory
from api.use_cases.batches import (
    BatchRequestRequeueReason,
    ProcessBatchRequestCommand,
    ProcessBatchRequestUseCase,
    ProcessBatchRequestUseCaseFailed,
    ProcessBatchRequestUseCaseRequeued,
    ProcessBatchRequestUseCaseSuccess,
)

REQUEST_ID = uuid4().hex
CAPACITY_SHARE = 0.5
DEFAULT_CONCURRENCY = 3
PARAMS = {"model": "chat-router", "messages": [{"role": "user", "content": "Hello"}]}


@pytest.fixture
def mock_model_tokenizer():
    tokenizer = create_autospec(ModelTokenizer, instance=True, spec_set=True)
    tokenizer.compute_tokens.side_effect = lambda texts: len(texts)
    return tokenizer


@pytest.fixture
def mock_model_environmental_impacts_computer():
    computer = create_autospec(ModelEnvironmentalImpactsComputer, instance=True, spec_set=True)
    computer.compute.return_value = EnvironmentalImpacts()
    return computer


@pytest.fixture
def mock_usage_context():
    usage_context = create_autospec(UsageContext, instance=True, spec_set=True)
    usage_context.get_request_id.return_value = REQUEST_ID
    usage_context.elapsed_ms.return_value = 500
    return usage_context


@pytest.fixture
def mock_random_generator():
    random_generator = create_autospec(random.Random, instance=True, spec_set=True)
    random_generator.sample.side_effect = lambda population, k: list(population)[:k]
    return random_generator


@pytest.fixture
def router():
    return RouterFactory(
        id=1, name="chat-router", type=RouterType.TEXT_GENERATION, providers=1, load_balancing_strategy=RouterLoadBalancingStrategy.SHUFFLE
    )


@pytest.fixture
def provider():
    return ProviderFactory(id=7, router_id=1, type=ProviderType.VLLM, qos_limit=8)


@pytest.fixture
def use_case(mock_model_tokenizer, mock_model_environmental_impacts_computer, mock_usage_context, mock_random_generator, router, provider):
    use_case = ProcessBatchRequestUseCase(
        model_environmental_impacts_computer=mock_model_environmental_impacts_computer,
        model_tokenizer=mock_model_tokenizer,
        provider_client=create_autospec(ProviderClient, instance=True, spec_set=True),
        provider_concurrency_limiter=create_autospec(ProviderConcurrencyLimiter, instance=True, spec_set=True),
        provider_repository=create_autospec(ProviderRepository, instance=True, spec_set=True),
        router_rate_limiter=create_autospec(RouterRateLimiter, instance=True, spec_set=True),
        router_repository=create_autospec(RouterRepository, instance=True, spec_set=True),
        usage_context=mock_usage_context,
        usage_repository=create_autospec(UsageRepository, instance=True, spec_set=True),
        random_generator=mock_random_generator,
        batch_repository=create_autospec(BatchRepository, instance=True, spec_set=True),
        provider_capacity_share=CAPACITY_SHARE,
        provider_default_concurrency=DEFAULT_CONCURRENCY,
    )
    use_case.router_repository.get_router_by_name_or_alias.return_value = router
    use_case.provider_repository.get_all_providers_of_router.return_value = [provider]
    use_case.provider_concurrency_limiter.reserve.return_value = ProviderReservation(provider=provider, request_id=REQUEST_ID)
    return use_case


def _command(attempts: int = 0) -> ProcessBatchRequestCommand:
    batch_request = ClaimedBatchRequest(
        id=11,
        batch_id=3,
        custom_id="request-1",
        position=0,
        params=PARAMS,
        status=BatchRequestStatus.PROCESSING,
        attempts=attempts,
        user_id=1,
        key_id=2,
    )
    return ProcessBatchRequestCommand(
        payload=PARAMS, authenticated_user=AuthenticatedUserFactory(id=1, admin=True), authenticated_key=KeyFactory(id=2), batch_request=batch_request
    )


def _chat_completion() -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-1",
        object="chat.completion",
        created=0,
        model="chat-router",
        choices=[{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )


class TestProcessBatchRequestUseCase:
    @pytest.mark.asyncio
    async def test_should_forward_the_request_and_store_the_completion(self, use_case, provider):
        # Arrange
        use_case.provider_client.forward.return_value = ProviderResponse(data=_chat_completion())

        # Act
        result = await use_case.execute(_command())

        # Assert
        assert result == ProcessBatchRequestUseCaseSuccess(request_id=11)
        use_case.provider_concurrency_limiter.reserve.assert_awaited_once_with(request_id=REQUEST_ID, provider=provider, limit=4)
        forwarded_request = use_case.provider_client.forward.await_args.kwargs["request"]
        assert forwarded_request == ProviderRequest(id=REQUEST_ID, endpoint=ProviderEndpoint.CHAT_COMPLETIONS, payload=forwarded_request.payload)
        assert forwarded_request.payload.messages == PARAMS["messages"]
        use_case.provider_concurrency_limiter.release.assert_awaited_once()
        use_case.usage_repository.save_record.assert_called_once()
        stored = use_case.batch_repository.complete_request.await_args.kwargs
        assert stored["request_id"] == 11
        assert stored["status"] == BatchRequestStatus.SUCCEEDED
        assert stored["response"]["id"] == "chatcmpl-1"
        assert stored["response"]["usage"]["prompt_tokens"] == 10
        use_case.router_rate_limiter.update_rate_limit_state.assert_not_called()

    @pytest.mark.asyncio
    async def test_should_reserve_the_default_concurrency_on_a_provider_without_qos_limit(self, use_case):
        # Arrange
        provider = ProviderFactory(id=8, router_id=1, qos_limit=None)
        use_case.provider_repository.get_all_providers_of_router.return_value = [provider]
        use_case.provider_concurrency_limiter.reserve.return_value = ProviderReservationRefused()

        # Act
        await use_case.execute(_command())

        # Assert
        use_case.provider_concurrency_limiter.reserve.assert_awaited_once_with(request_id=REQUEST_ID, provider=provider, limit=DEFAULT_CONCURRENCY)

    @pytest.mark.asyncio
    async def test_should_complete_as_errored_when_the_model_is_not_usable(self, use_case):
        # Arrange
        use_case.router_repository.get_router_by_name_or_alias.return_value = RouterNotFoundError(name="chat-router")

        # Act
        result = await use_case.execute(_command())

        # Assert
        error = BatchRequestError(type="ModelNotFoundError", message="Model chat-router not found.", status_code=404)
        assert result == ProcessBatchRequestUseCaseFailed(request_id=11, error=error)
        use_case.batch_repository.complete_request.assert_awaited_once_with(
            request_id=11, status=BatchRequestStatus.ERRORED, response=None, error=error
        )
        use_case.provider_client.forward.assert_not_awaited()

    @pytest.mark.parametrize("reservation", [ProviderReservationRefused(), ProviderLimitsCannotBeEnforced()])
    @pytest.mark.asyncio
    async def test_should_requeue_without_counting_an_attempt_when_no_provider_has_spare_capacity(self, use_case, reservation):
        # Arrange
        use_case.provider_concurrency_limiter.reserve.return_value = reservation

        # Act
        result = await use_case.execute(_command())

        # Assert
        assert result == ProcessBatchRequestUseCaseRequeued(request_id=11, reason=BatchRequestRequeueReason.NO_CAPACITY)
        use_case.batch_repository.requeue_request.assert_awaited_once_with(
            request_id=11, delay_seconds=ProcessBatchRequestUseCase.NO_CAPACITY_DELAY_SECONDS, count_attempt=False
        )
        use_case.provider_client.forward.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_should_retry_later_a_transient_provider_failure(self, use_case):
        # Arrange
        use_case.provider_client.forward.return_value = StatusCodeModelError(status_code=503, detail="Overloaded")

        # Act
        result = await use_case.execute(_command(attempts=1))

        # Assert
        assert result == ProcessBatchRequestUseCaseRequeued(request_id=11, reason=BatchRequestRequeueReason.RETRY)
        use_case.batch_repository.requeue_request.assert_awaited_once_with(
            request_id=11, delay_seconds=ProcessBatchRequestUseCase.RETRY_DELAY_SECONDS * 2, count_attempt=True
        )
        use_case.provider_concurrency_limiter.release.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_should_complete_as_errored_a_transient_failure_on_the_last_attempt(self, use_case):
        # Arrange
        use_case.provider_client.forward.return_value = TooBusyModelError(status_code=500, detail="ReadTimeout")

        # Act
        result = await use_case.execute(_command(attempts=ProcessBatchRequestUseCase.MAX_ATTEMPTS - 1))

        # Assert
        assert result == ProcessBatchRequestUseCaseFailed(
            request_id=11, error=BatchRequestError(type="TooBusyModelError", message="ReadTimeout", status_code=500)
        )
        use_case.batch_repository.requeue_request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_should_complete_as_errored_a_request_the_provider_rejects(self, use_case):
        # Arrange
        use_case.provider_client.forward.return_value = StatusCodeModelError(status_code=400, detail="max_tokens is too large")

        # Act
        result = await use_case.execute(_command())

        # Assert
        assert result == ProcessBatchRequestUseCaseFailed(
            request_id=11, error=BatchRequestError(type="StatusCodeModelError", message="max_tokens is too large", status_code=400)
        )
        use_case.batch_repository.requeue_request.assert_not_awaited()
        use_case.usage_repository.save_record.assert_not_called()
