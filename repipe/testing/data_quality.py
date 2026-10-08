import functools
from enum import StrEnum
from typing import Callable, Any, cast

from pydantic import BaseModel

type _TestInput = Any


class FailType(StrEnum):
    WARN = "WARN"
    """ Does not stop job if a node fails. Though will stop descendents from failing node(s)."""

    FAIL = "FAIL"
    """ Stops the entire job if a node fails. """


class TestResult(BaseModel):
    success: bool = False


type DataQualityTest = Callable[[_TestInput], TestResult]


class RegisteredDataQualityTest(BaseModel):
    """ Pairs a decorated test function with the settings it was declared with. """
    model_config = {"arbitrary_types_allowed": True}

    name: str
    func: DataQualityTest
    fail_type: FailType


def data_quality_test(
        func: DataQualityTest | None = None,
        *,
        fail_type: FailType = FailType.FAIL
):
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
