from __future__ import annotations

import numpy as np
import pytest

from textdistance._lib import address


@pytest.mark.parametrize("dtype", [np.uint32, np.int64, np.float64])
def test_address_accepts_native_contiguous_writable_buffers(dtype):
    array = np.zeros(4, dtype=dtype)
    assert address(array) == array.ctypes.data


def test_address_rejects_wrong_dtype():
    with pytest.raises(TypeError, match="dtype"):
        address(np.zeros(4, dtype=np.uint64))


def test_address_rejects_strided_buffer():
    with pytest.raises(ValueError, match="contiguous"):
        address(np.zeros(8, dtype=np.uint32)[::2])


def test_address_rejects_read_only_buffer():
    array = np.zeros(4, dtype=np.uint32)
    array.flags.writeable = False
    assert address(array) == array.ctypes.data
    with pytest.raises(ValueError, match="writable"):
        address(array, writable=True)
