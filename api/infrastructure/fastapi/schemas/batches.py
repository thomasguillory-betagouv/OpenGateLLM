from typing import Annotated, Any, Literal

from pydantic import Field, StringConstraints, field_validator, model_validator

from api.domain import BaseModel
from api.domain.batch.entities import BatchProcessingStatus, BatchRequestStatus
from api.infrastructure.fastapi.schemas import UnixTimestamp
from api.infrastructure.fastapi.schemas.chat import CreateChatCompletionsBody

MAX_BATCH_REQUESTS = 10_000


class BatchRequestBody(BaseModel):
    custom_id: Annotated[str, StringConstraints(pattern=r"^[a-zA-Z0-9_-]{1,64}$"), Field(description="Your identifier for the request, unique within the batch. Results are matched by custom_id: their order is not guaranteed.", examples=["request-1"])]  # fmt: off
    params: Annotated[CreateChatCompletionsBody, Field(description="The chat completion request, as sent to `POST /v1/chat/completions`. Streaming is not supported.")]  # fmt: off

    @field_validator("params", mode="after")
    def reject_stream(cls, params: CreateChatCompletionsBody) -> CreateChatCompletionsBody:
        if params.stream:
            raise ValueError("Streaming is not supported in a batch.")
        return params


class CreateBatchBody(BaseModel):
    requests: Annotated[list[BatchRequestBody], Field(min_length=1, max_length=MAX_BATCH_REQUESTS, description=f"The requests of the batch, at most {MAX_BATCH_REQUESTS}.")]  # fmt: off


class BatchRequestCountsResponse(BaseModel):
    processing: Annotated[int, Field(description="Number of requests not processed yet.")]
    succeeded: Annotated[int, Field(description="Number of requests that succeeded.")]
    errored: Annotated[int, Field(description="Number of requests that failed.")]
    canceled: Annotated[int, Field(description="Number of requests canceled before being processed.")]
    expired: Annotated[int, Field(description="Number of requests not processed before the batch expired.")]


class BatchResponse(BaseModel):
    object: Annotated[Literal["batch"], Field(default="batch", description="Type of the object.")]
    id: Annotated[int, Field(description="ID of the batch.")]
    processing_status: Annotated[BatchProcessingStatus, Field(description="Processing status of the batch. Its results are available once it is `ended`.")]  # fmt: off
    request_counts: Annotated[BatchRequestCountsResponse, Field(description="Number of requests of the batch by status.")]
    created: Annotated[UnixTimestamp, Field(description="Time of creation, as Unix timestamp.")]
    expires: Annotated[UnixTimestamp, Field(description="Time after which the requests not processed yet expire, as Unix timestamp.")]
    ended: Annotated[UnixTimestamp | None, Field(default=None, description="Time the processing ended, as Unix timestamp.")]
    cancel_initiated: Annotated[UnixTimestamp | None, Field(default=None, description="Time the cancellation was requested, as Unix timestamp.")]  # fmt: off


class BatchesResponse(BaseModel):
    object: Annotated[Literal["list"], Field(default="list", description="Type of the object.")]
    total: Annotated[int, Field(description="Total number of batches.")]
    offset: Annotated[int, Field(description="Offset of the batches list.")]
    limit: Annotated[int, Field(description="Limit of the batches list.")]
    data: Annotated[list[BatchResponse], Field(description="List of batches, newest first.")]


class BatchRequestErrorResponse(BaseModel):
    type: Annotated[str, Field(description="Type of the error.")]
    message: Annotated[str, Field(description="Description of the error.")]
    status_code: Annotated[int | None, Field(default=None, description="HTTP status the request would have received from `POST /v1/chat/completions`, when known.")]  # fmt: off


class BatchResultBody(BaseModel):
    type: Annotated[BatchRequestStatus, Field(description="Outcome of the request: `succeeded`, `errored`, `canceled` or `expired`.")]
    response: Annotated[dict[str, Any] | None, Field(default=None, description="The chat completion, when the request succeeded.")]
    error: Annotated[BatchRequestErrorResponse | None, Field(default=None, description="The error, when the request failed.")]


class BatchResultResponse(BaseModel):
    """One line of the results file."""

    custom_id: Annotated[str, Field(description="The custom_id of the request.")]
    result: Annotated[BatchResultBody, Field(description="The result of the request.")]

    @model_validator(mode="before")
    @classmethod
    def nest_the_outcome_under_result(cls, request: Any) -> Any:
        # the results file nests the outcome of the flat BatchRequest under result, as Anthropic's Message Batches API does
        if isinstance(request, dict) or not hasattr(request, "status"):
            return request
        error = request.error.model_dump() if request.error is not None else None
        return {"custom_id": request.custom_id, "result": {"type": request.status, "response": request.response, "error": error}}
