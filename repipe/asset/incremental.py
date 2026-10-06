from enum import StrEnum

from pydantic import BaseModel


class IncrementalStrategy(StrEnum):
    APPEND = "APPEND"
    """ Loads existing parquet, appends, writes back to disk """

    UPSERT = "UPSERT"
    """ Loads existing parquet, removes duped rows, writes back to disk """


class IncrementalSettings(BaseModel):
    date_column: str
    primary_key: str | list[str]
    strategy: IncrementalStrategy
