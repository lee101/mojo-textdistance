"""ctypes bridge to the compiled Mojo kernels."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCE = os.path.join(ROOT, "src", "textdistance.mojo")
LIBRARY = os.path.join(ROOT, "dist", "libmojo-textdistance.so")
BUILD_SCRIPT = os.path.join(ROOT, "build", "build.sh")

I = ctypes.c_int64
F = ctypes.c_double
_SIGNATURES = {
    "mtd_hamming": ([I] * 5, I),
    "mtd_levenshtein": ([I] * 5, I),
    "mtd_damerau_restricted": ([I] * 5, I),
    "mtd_damerau_unrestricted": ([I] * 7, I),
    "mtd_jaro": ([I] * 8 + [F], F),
    "mtd_lcsseq": ([I] * 5, I),
    "mtd_lcsstr": ([I] * 6, I),
    "mtd_prefix": ([I] * 4, I),
    "mtd_postfix": ([I] * 4, I),
    "mtd_token_stats": ([I] * 9, None),
    "mtd_needleman_wunsch": ([I] * 4 + [F, I], F),
    "mtd_smith_waterman": ([I] * 4 + [F, I], F),
}

_library: ctypes.CDLL | None = None


def build(force: bool = False) -> str:
    """Build the shared library when it is absent or older than the source."""
    if (
        not force
        and os.path.exists(LIBRARY)
        and os.path.getmtime(LIBRARY) >= os.path.getmtime(SOURCE)
    ):
        return LIBRARY
    proc = subprocess.run(
        ["bash", BUILD_SCRIPT],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode != 0 or not os.path.exists(LIBRARY):
        output = "\n".join(part.strip() for part in (proc.stdout, proc.stderr) if part.strip())
        raise RuntimeError(output[:6000] or "Mojo build failed without diagnostic output")
    return LIBRARY


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def address(array: np.ndarray, *, writable: bool = False) -> int:
    """Return an ABI-safe address while keeping validation close to the call site.

    The Mojo API accepts only the three native, fixed-width dtypes used by this
    package.  ctypes keeps ``array`` alive for the duration of this Python call;
    callers also retain their array local until the foreign call returns.
    """
    if not isinstance(array, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if array.dtype not in (np.dtype(np.uint32), np.dtype(np.int64), np.dtype(np.float64)):
        raise TypeError(f"unsupported FFI dtype: {array.dtype}")
    if not array.flags.c_contiguous:
        raise ValueError("FFI buffers must be C-contiguous")
    if not array.flags.aligned:
        raise ValueError("FFI buffers must be aligned")
    if writable and not array.flags.writeable:
        raise ValueError("FFI buffers must be writable")
    pointer = int(array.ctypes.data)
    if array.size and pointer == 0:
        raise ValueError("non-empty FFI buffer has a null pointer")
    if array.size > ctypes.c_int64(-1).value + 2**63:
        raise OverflowError("FFI buffer is too large for a signed 64-bit length")
    return pointer
