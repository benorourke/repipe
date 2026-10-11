from typing import Callable

from pydantic import BaseModel

type PartitionType = str


class PartitionSettings(BaseModel):
    partition_columns: list[str]
    make_partitions: Callable[[], PartitionType]
