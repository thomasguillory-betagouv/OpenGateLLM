from enum import StrEnum
from typing import Any

from api.domain import BaseModel, EntitiesPage, UtcDatetime


class BatchProcessingStatus(StrEnum):
    IN_PROGRESS = "in_progress"
    CANCELING = "canceling"
    ENDED = "ended"


class BatchRequestStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    ERRORED = "errored"
    CANCELED = "canceled"
    EXPIRED = "expired"


class BatchRequestCounts(BaseModel):
    processing: int = 0
    succeeded: int = 0
    errored: int = 0
    canceled: int = 0
    expired: int = 0


class Batch(BaseModel):
    id: int
    user_id: int
    key_id: int
    processing_status: BatchProcessingStatus
    request_counts: BatchRequestCounts
    created: UtcDatetime
    expires: UtcDatetime
    ended: UtcDatetime | None = None
    cancel_initiated: UtcDatetime | None = None

    @property
    def has_ended(self) -> bool:
        return self.processing_status == BatchProcessingStatus.ENDED

    @property
    def is_canceling(self) -> bool:
        return self.processing_status == BatchProcessingStatus.CANCELING


BatchPage = EntitiesPage["Batch"]


class BatchRequestToCreate(BaseModel):
    custom_id: str
    params: dict[str, Any]


class BatchRequestError(BaseModel):
    type: str
    message: str
    status_code: int | None = None


class BatchRequest(BaseModel):
    id: int
    batch_id: int
    custom_id: str
    position: int
    params: dict[str, Any]
    status: BatchRequestStatus
    attempts: int
    response: dict[str, Any] | None = None
    error: BatchRequestError | None = None


class ClaimedBatchRequest(BatchRequest):
    """A request a worker holds until it completes or requeues it. It carries the owner of its batch, whose access
    is checked again when the request is processed."""

    user_id: int
    key_id: int
