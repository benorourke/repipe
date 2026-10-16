from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

import polars as pl
from pydantic import BaseModel, field_validator, model_validator

type PartitionValue = str | int | date | bool

# Polars writes null keys to this directory name, so a string with this value would collide with null.
RESERVED_VALUE = "__HIVE_DEFAULT_PARTITION__"


class PartitionSettings(BaseModel):
    """ Splits an asset on disk into one directory per distinct value of `columns`, nested in
    `columns` order (`date=2026-10-12/track=ascot/`). Keys come from the data the asset emits.

    `overwrite_by` sets what `Mode.OVERWRITE` replaces, and must be a prefix of `columns`:
    `[]` (default) replaces the whole asset, same as an unpartitioned asset, so a run that only
    emits today's dates deletes every other date. `["date"]` replaces only the dates a run
    wrote to. """
    model_config = {"arbitrary_types_allowed": True}

    columns: dict[str, Any]
    """ Partition column name -> polars dtype. Used as the scan's `hive_schema`. """
    overwrite_by: list[str] = []
    max_partitions_per_run: int = 1000

    @field_validator("columns")
    @classmethod
    def _check_columns(cls, columns: dict[str, Any]) -> dict[str, pl.DataType]:
        if not columns:
            raise ValueError("PartitionSettings.columns must name at least one column")
        normalized = {}
        for name, dtype in columns.items():
            dtype = dtype() if isinstance(dtype, type) else dtype
            if not isinstance(dtype, pl.DataType) or not _is_allowed_dtype(dtype):
                # Floats are ambiguous as paths (1.0 vs 1); datetimes don't round-trip timezones.
                raise TypeError(
                    f"partition column {name!r} has dtype {dtype!r} - use String, an integer type, Date or Boolean"
                )
            normalized[name] = dtype
        return normalized

    @model_validator(mode="after")
    def _check_overwrite_by_is_a_prefix(self) -> PartitionSettings:
        if self.overwrite_by != list(self.columns)[:len(self.overwrite_by)]:
            raise ValueError(
                f"overwrite_by {self.overwrite_by!r} must be a prefix of the partition columns {list(self.columns)!r}"
            )
        return self

    @property
    def hive_schema(self) -> dict[str, pl.DataType]:
        return dict(self.columns)


def _is_allowed_dtype(dtype: pl.DataType) -> bool:
    return dtype == pl.String or dtype == pl.Date or dtype == pl.Boolean or dtype.is_integer()


class PartitionKey(Mapping[str, PartitionValue]):
    """ One partition's key values, ordered like `PartitionSettings.columns`. Hashable, so it
    can key a dict. A prefix of the columns is also a valid key (e.g. a replace unit). """

    __slots__ = ("_items",)

    def __init__(self, items: Iterable[tuple[str, PartitionValue]]):
        self._items = tuple(items)

    @classmethod
    def from_mapping(cls, settings: PartitionSettings, values: Mapping[str, Any]) -> PartitionKey:
        """ Validates a user-supplied key, e.g. from `Output(partition=...)`. """
        if set(values) != set(settings.columns):
            raise ValueError(f"partition {dict(values)!r} must have exactly the columns {list(settings.columns)!r}")
        for name, dtype in settings.columns.items():
            if not _matches_dtype(values[name], dtype):
                raise TypeError(f"partition value {values[name]!r} for {name!r} doesn't match its dtype {dtype!r}")
        return cls((name, values[name]) for name in settings.columns)

    @classmethod
    def from_relative_path(cls, settings: PartitionSettings, path: Path) -> PartitionKey | None:
        """ The inverse of `relative_path`, or `None` if `path` isn't a leaf partition directory
        for these settings, e.g. it was written before a column's dtype changed. """
        if len(path.parts) != len(settings.columns):
            return None
        items = []
        for part, (name, dtype) in zip(path.parts, settings.columns.items()):
            column, sep, encoded = part.partition("=")
            if column != name or not sep:
                return None
            try:
                value = _decode(unquote(encoded), dtype)
            except ValueError:
                return None
            # Rejects values that parse but aren't how we'd write them, e.g. "007" as an integer.
            if encode_value(value) != encoded:
                return None
            items.append((name, value))
        return cls(items)

    def relative_path(self) -> Path:
        return Path(*(f"{name}={encode_value(value)}" for name, value in self._items))

    def prefix(self, length: int) -> PartitionKey:
        return PartitionKey(self._items[:length])

    def __getitem__(self, name: str) -> PartitionValue:
        for key, value in self._items:
            if key == name:
                return value
        raise KeyError(name)

    def __iter__(self) -> Iterator[str]:
        return (name for name, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __hash__(self) -> int:
        return hash(self._items)

    def __repr__(self) -> str:
        return f"PartitionKey({dict(self._items)!r})"


def format_value(value: PartitionValue | None) -> str:
    """ A key value as written in a directory name, before percent-encoding. """
    if value is None:
        return RESERVED_VALUE
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def encode_value(value: PartitionValue | None) -> str:
    return quote(format_value(value), safe="")


def _decode(text: str, dtype: pl.DataType) -> PartitionValue:
    if dtype == pl.Date:
        return date.fromisoformat(text)
    if dtype == pl.Boolean:
        if text not in ("true", "false"):
            raise ValueError(f"not a boolean: {text!r}")
        return text == "true"
    if dtype.is_integer():
        return int(text)
    return text


def _matches_dtype(value: object, dtype: pl.DataType) -> bool:
    if dtype == pl.Date:
        return isinstance(value, date) and not isinstance(value, datetime)
    if dtype == pl.Boolean:
        return isinstance(value, bool)
    if dtype.is_integer():
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, str)
