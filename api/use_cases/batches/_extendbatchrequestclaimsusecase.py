from dataclasses import dataclass

from api.domain.batch import BatchRepository


@dataclass
class ExtendBatchRequestClaimsCommand:
    claim_ids: list[str]
    lease_seconds: int


class ExtendBatchRequestClaimsUseCase:
    """The worker's heartbeat: a claim stays valid as long as its worker is alive, whatever the provider timeout, and
    expires shortly after the worker dies."""

    def __init__(self, batch_repository: BatchRepository):
        self.batch_repository = batch_repository

    async def execute(self, command: ExtendBatchRequestClaimsCommand) -> None:
        await self.batch_repository.extend_claims(claim_ids=command.claim_ids, lease_seconds=command.lease_seconds)
