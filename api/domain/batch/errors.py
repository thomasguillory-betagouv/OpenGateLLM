from dataclasses import dataclass


@dataclass
class BatchNotFoundError:
    id: int


@dataclass
class BatchAlreadyEndedError:
    id: int


@dataclass
class BatchNotEndedError:
    id: int


@dataclass
class BatchCustomIdNotUniqueError:
    custom_ids: list[str]
