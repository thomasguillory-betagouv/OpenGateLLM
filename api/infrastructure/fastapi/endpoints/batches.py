from collections.abc import AsyncGenerator
from contextlib import aclosing

from fastapi import APIRouter, Body, Depends, Path, Query, Security
from fastapi.responses import StreamingResponse

from api.dependencies import (
    cancel_batch_use_case_factory,
    create_batch_use_case_factory,
    get_batch_results_use_case_factory,
    get_batches_use_case_factory,
    get_one_batch_use_case_factory,
)
from api.domain.batch.entities import BatchRequest, BatchRequestToCreate
from api.domain.batch.errors import BatchAlreadyEndedError, BatchCustomIdNotUniqueError, BatchNotEndedError, BatchNotFoundError
from api.domain.key.entities import Key
from api.domain.router.errors import RouterHasWrongTypeError, RouterNotFoundError
from api.domain.user.views import AuthenticatedUserView
from api.infrastructure.fastapi.accesscontroller import AccessController
from api.infrastructure.fastapi.dependencies import get_authenticated_key, get_authenticated_user
from api.infrastructure.fastapi.documentation import get_documentation_responses
from api.infrastructure.fastapi.endpoints.exceptions import (
    BatchAlreadyEndedHTTPException,
    BatchCustomIdNotUniqueHTTPException,
    BatchNotEndedHTTPException,
    BatchNotFoundHTTPException,
    ModelNotFoundHTTPException,
    WrongModelTypeHTTPException,
)
from api.infrastructure.fastapi.routes import EndpointRoute, RouterName
from api.infrastructure.fastapi.schemas.batches import BatchesResponse, BatchResponse, BatchResultResponse, CreateBatchBody
from api.use_cases.batches import (
    CancelBatchCommand,
    CancelBatchUseCase,
    CancelBatchUseCaseSuccess,
    CreateBatchCommand,
    CreateBatchUseCase,
    CreateBatchUseCaseSuccess,
    GetBatchesCommand,
    GetBatchesUseCase,
    GetBatchesUseCaseSuccess,
    GetBatchResultsCommand,
    GetBatchResultsUseCase,
    GetBatchResultsUseCaseSuccess,
    GetOneBatchCommand,
    GetOneBatchUseCase,
    GetOneBatchUseCaseSuccess,
)

router = APIRouter(prefix="/v1", tags=[RouterName.BATCHES.title()])


async def _as_jsonl_lines(results: AsyncGenerator[BatchRequest]) -> AsyncGenerator[str]:
    async with aclosing(results):
        async for result in results:
            yield BatchResultResponse.model_validate(result, from_attributes=True).model_dump_json() + "\n"


@router.post(
    path=EndpointRoute.CHAT_COMPLETIONS_BATCHES,
    dependencies=[Security(dependency=AccessController())],
    status_code=201,
    responses=get_documentation_responses([BatchCustomIdNotUniqueHTTPException, ModelNotFoundHTTPException, WrongModelTypeHTTPException]),
)
async def create_batch(
    body: CreateBatchBody = Body(description="The batch creation request."),
    create_batch_use_case: CreateBatchUseCase = Depends(create_batch_use_case_factory),
    authenticated_user: AuthenticatedUserView = Depends(get_authenticated_user),
    authenticated_key: Key = Depends(get_authenticated_key),
) -> BatchResponse:
    """
    Create a batch of chat completion requests, processed asynchronously within 24 hours.

    Batch requests only use the capacity interactive traffic leaves free. Poll the batch until its `processing_status`
    is `ended`, then download its results.
    """
    command = CreateBatchCommand(
        authenticated_user=authenticated_user,
        authenticated_key=authenticated_key,
        requests=[BatchRequestToCreate(custom_id=request.custom_id, params=request.params.model_dump()) for request in body.requests],
    )
    result = await create_batch_use_case.execute(command)

    match result:
        case CreateBatchUseCaseSuccess(batch=batch):
            return BatchResponse.model_validate(batch, from_attributes=True)
        case BatchCustomIdNotUniqueError(custom_ids=custom_ids):
            raise BatchCustomIdNotUniqueHTTPException(custom_ids=custom_ids)
        case RouterNotFoundError(name=name):
            raise ModelNotFoundHTTPException(name=name)
        case RouterHasWrongTypeError(actual_type=actual_type, expected_type=expected_type):
            raise WrongModelTypeHTTPException(actual_type=actual_type, expected_type=expected_type)


@router.get(
    path=EndpointRoute.CHAT_COMPLETIONS_BATCHES,
    dependencies=[Security(dependency=AccessController())],
    status_code=200,
    responses=get_documentation_responses([]),
)
async def get_batches(
    offset: int = Query(default=0, ge=0, description="Number of batches to skip."),
    limit: int = Query(default=10, ge=1, le=100, description="Maximum number of batches to return."),
    get_batches_use_case: GetBatchesUseCase = Depends(get_batches_use_case_factory),
    authenticated_user: AuthenticatedUserView = Depends(get_authenticated_user),
) -> BatchesResponse:
    """
    Get your batches, newest first.
    """
    result = await get_batches_use_case.execute(GetBatchesCommand(user_id=authenticated_user.id, offset=offset, limit=limit))

    match result:
        case GetBatchesUseCaseSuccess(batch_page=batch_page):
            return BatchesResponse(
                total=batch_page.total,
                offset=offset,
                limit=limit,
                data=[BatchResponse.model_validate(batch, from_attributes=True) for batch in batch_page.data],
            )


@router.get(
    path=EndpointRoute.CHAT_COMPLETIONS_BATCHES + "/{batch_id}",
    dependencies=[Security(dependency=AccessController())],
    status_code=200,
    responses=get_documentation_responses([BatchNotFoundHTTPException]),
)
async def get_batch(
    batch_id: int = Path(description="The ID of the batch."),
    get_one_batch_use_case: GetOneBatchUseCase = Depends(get_one_batch_use_case_factory),
    authenticated_user: AuthenticatedUserView = Depends(get_authenticated_user),
) -> BatchResponse:
    """
    Get one of your batches.
    """
    result = await get_one_batch_use_case.execute(GetOneBatchCommand(batch_id=batch_id, user_id=authenticated_user.id))

    match result:
        case GetOneBatchUseCaseSuccess(batch=batch):
            return BatchResponse.model_validate(batch, from_attributes=True)
        case BatchNotFoundError(id=not_found_batch_id):
            raise BatchNotFoundHTTPException(batch_id=not_found_batch_id)


@router.post(
    path=EndpointRoute.CHAT_COMPLETIONS_BATCHES + "/{batch_id}/cancel",
    dependencies=[Security(dependency=AccessController())],
    status_code=200,
    responses=get_documentation_responses([BatchNotFoundHTTPException, BatchAlreadyEndedHTTPException]),
)
async def cancel_batch(
    batch_id: int = Path(description="The ID of the batch."),
    cancel_batch_use_case: CancelBatchUseCase = Depends(cancel_batch_use_case_factory),
    authenticated_user: AuthenticatedUserView = Depends(get_authenticated_user),
) -> BatchResponse:
    """
    Cancel one of your batches. Its pending requests are canceled; those already processing run to the end, then the
    batch ends.
    """
    result = await cancel_batch_use_case.execute(CancelBatchCommand(batch_id=batch_id, user_id=authenticated_user.id))

    match result:
        case CancelBatchUseCaseSuccess(batch=batch):
            return BatchResponse.model_validate(batch, from_attributes=True)
        case BatchNotFoundError(id=not_found_batch_id):
            raise BatchNotFoundHTTPException(batch_id=not_found_batch_id)
        case BatchAlreadyEndedError(id=ended_batch_id):
            raise BatchAlreadyEndedHTTPException(batch_id=ended_batch_id)


@router.get(
    path=EndpointRoute.CHAT_COMPLETIONS_BATCHES + "/{batch_id}/results",
    dependencies=[Security(dependency=AccessController())],
    status_code=200,
    responses=get_documentation_responses([BatchNotFoundHTTPException, BatchNotEndedHTTPException]),
    response_class=StreamingResponse,
)
async def get_batch_results(
    batch_id: int = Path(description="The ID of the batch."),
    get_batch_results_use_case: GetBatchResultsUseCase = Depends(get_batch_results_use_case_factory),
    authenticated_user: AuthenticatedUserView = Depends(get_authenticated_user),
) -> StreamingResponse:
    """
    Download the results of an ended batch, as JSON Lines: one `{"custom_id", "result"}` object per request.
    """
    result = await get_batch_results_use_case.execute(GetBatchResultsCommand(batch_id=batch_id, user_id=authenticated_user.id))

    match result:
        case GetBatchResultsUseCaseSuccess(results=results):
            return StreamingResponse(content=_as_jsonl_lines(results), media_type="application/x-ndjson")
        case BatchNotFoundError(id=not_found_batch_id):
            raise BatchNotFoundHTTPException(batch_id=not_found_batch_id)
        case BatchNotEndedError(id=not_ended_batch_id):
            raise BatchNotEndedHTTPException(batch_id=not_ended_batch_id)
