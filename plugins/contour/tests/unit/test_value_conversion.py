import numpy as np
import pytest

from contour.value_conversion import to_float, to_int

pytestmark = pytest.mark.fast


@pytest.mark.parametrize(
    "value", [0, -3, 2.7, "12", b"12", bytearray(b"12"), memoryview(b"12"), np.int64(12), np.float32(2.7)]
)
def test_numeric_conversions_match_builtins(value: object) -> None:
    assert to_int(value) == int(value)
    assert to_float(value) == float(value)


@pytest.mark.parametrize("value", [None, [], {}, object()])
def test_unsupported_numeric_values_raise(value: object) -> None:
    with pytest.raises(TypeError):
        to_int(value)
    with pytest.raises(TypeError):
        to_float(value)


def test_invalid_numeric_text_is_not_replaced_with_a_default() -> None:
    with pytest.raises(ValueError):
        to_int("invalid")
    with pytest.raises(ValueError):
        to_float("invalid")
