import functools
from enum import StrEnum
from typing import Any, Callable, cast, overload

import duckdb
import polars as pl
from pydantic import BaseModel

SAMPLE_ROWS = 5
""" How many offending rows a result keeps, so the report can show what broke without printing a whole table. """


class FailType(StrEnum):
    WARN = "WARN"
    """ Reported, but the job carries on. """

    FAIL = "FAIL"
    """ Stops the entire job, mid-run, as soon as the test fails. """


class DataQualityStatus(StrEnum):
    """ How a test ended up: it passed, or failed at the severity its `FailType` gives it. """
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


type DataQualityReturn = pl.DataFrame | pl.LazyFrame | duckdb.DuckDBPyRelation | None
"""" Offending rows or None. """

type DataQualityTest = Callable[..., DataQualityReturn]


class DataQualityResult(BaseModel):
    """ How one test went: it passes unless it raised or returned offending rows. """
    model_config = {"arbitrary_types_allowed": True}

    test_id: str
    fail_type: FailType
    offending_count: int = 0
    sample: pl.DataFrame | None = None
    """ The first `SAMPLE_ROWS` offending rows. """
    error: str | None = None
    """ Set when the test raised, or returned something it shouldn't have. """

    @property
    def passed(self) -> bool:
        return self.error is None and self.offending_count == 0

    @property
    def status(self) -> DataQualityStatus:
        return DataQualityStatus.PASS if self.passed else DataQualityStatus(self.fail_type)

    @classmethod
    def from_return(cls, test_id: str, fail_type: FailType, value: object) -> "DataQualityResult":
        """ Reads a test's return value: `None` passes, any frame is the offending rows. """
        if value is None:
            return cls(test_id=test_id, fail_type=fail_type)
        if isinstance(value, duckdb.DuckDBPyRelation):
            value = value.pl()
        if isinstance(value, pl.LazyFrame):
            value = value.collect()
        if not isinstance(value, pl.DataFrame):
            return cls.from_error(
                test_id, fail_type,
                f"returned {type(value).__name__!r} - return the offending rows as a DataFrame, or None",
            )
        return cls(
            test_id=test_id, fail_type=fail_type,
            offending_count=value.height, sample=value.head(SAMPLE_ROWS) if value.height else None,
        )

    @classmethod
    def from_error(cls, test_id: str, fail_type: FailType, error: BaseException | str) -> "DataQualityResult":
        message = error if isinstance(error, str) else f"{type(error).__name__}: {error}"
        return cls(test_id=test_id, fail_type=fail_type, error=message)

    def describe(self) -> str:
        """ The line printed for this result, followed by a sample of the offending rows when there are any. """
        line = f"{self.status.value:<4} {self.test_id}"
        if self.error is not None:
            return f"{line}\n     {self.error}"
        if self.passed:
            return line
        return f"{line}  ({self.offending_count} offending rows)\n{self.sample}"


class DataQualityError(Exception):
    """ A `FailType.FAIL` test failed, so the job stops. Carries every result so far, the failing one last. """

    def __init__(self, results: list[DataQualityResult]):
        self.results = results
        super().__init__(f"data quality test {results[-1].test_id!r} failed")


class RegisteredDataQualityTest(BaseModel):
    """ Pairs a decorated test function with the settings it was declared with. """
    model_config = {"arbitrary_types_allowed": True}

    name: str
    func: DataQualityTest
    fail_type: FailType


@overload
def data_quality_test(func: DataQualityTest) -> DataQualityTest: ...


@overload
def data_quality_test(func: None = None, *, fail_type: FailType = FailType.FAIL) -> Callable[
    [DataQualityTest], DataQualityTest]: ...


def data_quality_test(
        func: DataQualityTest | None = None,
        *,
        fail_type: FailType = FailType.FAIL
) -> DataQualityTest | Callable[[DataQualityTest], DataQualityTest]:
    """ Supports both bare `@data_quality_test` and parameterized `@data_quality_test(...)` usage. """

    def decorator(inner_func: DataQualityTest) -> DataQualityTest:
        @functools.wraps(inner_func)
        def inner(*args, **kwargs):
            return inner_func(*args, **kwargs)

        registered = RegisteredDataQualityTest(
            name=inner_func.__name__,
            func=inner_func,
            fail_type=fail_type,
        )
        cast(Any, inner).__data_quality_test__ = registered

        return inner

    if func is not None:
        return decorator(func)

    return decorator
