"""Hand-rolled safetensors reader.

The `safetensors` library is deliberately not used. The format is simple enough
to parse directly, and parsing it yourself is the point of phase 1.

Layout on disk:

    [0:8]      u64 little-endian  — length N of the JSON header
    [8:8+N]    N bytes            — UTF-8 JSON header
    [8+N:]     raw tensor bytes   — data_offsets in the header are relative
                                     to the start of THIS region, not the file

The header maps tensor name -> {"dtype", "shape", "data_offsets": [start, end]}
plus an optional "__metadata__" key holding free-form strings.
"""

from __future__ import annotations

import json
import mmap
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# safetensors dtype tag -> (numpy dtype to read as, itemsize in bytes)
# bf16 has no numpy equivalent, so it is read as raw u16 and widened by hand.
_DTYPES: dict[str, tuple[np.dtype, int]] = {
    "F64": (np.dtype("<f8"), 8),
    "F32": (np.dtype("<f4"), 4),
    "F16": (np.dtype("<f2"), 2),
    "BF16": (np.dtype("<u2"), 2),
    "I64": (np.dtype("<i8"), 8),
    "I32": (np.dtype("<i4"), 4),
    "I16": (np.dtype("<i2"), 2),
    "I8": (np.dtype("<i1"), 1),
    "U8": (np.dtype("<u1"), 1),
    "BOOL": (np.dtype("?"), 1),
}


def bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    """Widen bfloat16 (stored as u16) to float32.

    bfloat16 is literally the top 16 bits of a float32 — same 8 exponent bits,
    mantissa truncated from 23 to 7. So widening is a left shift by 16, with no
    rounding and no special-case handling: inf, nan and zero all carry over
    because the exponent field is untouched.
    """
    return (raw.astype(np.uint32) << 16).view(np.float32)


@dataclass(frozen=True)
class TensorInfo:
    """Where one tensor lives in the file, before it is read."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    begin: int  # byte offset within the data region
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.begin

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


class SafeTensors:
    """Lazily-read safetensors file. Tensors are mmap'd, not loaded up front."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._f = open(self.path, "rb")
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)

        header_len = int.from_bytes(self._mm[0:8], "little")
        raw_header = self._mm[8 : 8 + header_len]
        header = json.loads(raw_header.decode("utf-8"))

        self.metadata: dict[str, str] = header.pop("__metadata__", {})
        self._data_start = 8 + header_len

        self.tensors: dict[str, TensorInfo] = {}
        for name, spec in header.items():
            begin, end = spec["data_offsets"]
            self.tensors[name] = TensorInfo(
                name=name,
                dtype=spec["dtype"],
                shape=tuple(spec["shape"]),
                begin=begin,
                end=end,
            )

        self._validate()

    def _validate(self) -> None:
        """Every tensor's byte span must match shape x itemsize exactly.

        A mismatch means either a corrupt file or a dtype we are decoding wrong,
        and both produce silently garbled weights rather than an error later.
        """
        for t in self.tensors.values():
            if t.dtype not in _DTYPES:
                raise ValueError(f"{t.name}: unsupported dtype {t.dtype!r}")
            _, itemsize = _DTYPES[t.dtype]
            expected = t.n_elements * itemsize
            if t.nbytes != expected:
                raise ValueError(
                    f"{t.name}: header says {t.nbytes} bytes but "
                    f"{t.shape} x {itemsize}B = {expected}"
                )

    def __contains__(self, name: str) -> bool:
        return name in self.tensors

    def __len__(self) -> int:
        return len(self.tensors)

    def keys(self):
        return self.tensors.keys()

    def get(self, name: str) -> np.ndarray:
        """Read one tensor as float32 (or its native type, if not floating)."""
        t = self.tensors[name]
        np_dtype, _ = _DTYPES[t.dtype]

        start = self._data_start + t.begin
        buf = self._mm[start : self._data_start + t.end]
        arr = np.frombuffer(buf, dtype=np_dtype)

        if t.dtype == "BF16":
            arr = bf16_to_f32(arr)
        elif t.dtype == "F16":
            arr = arr.astype(np.float32)

        return arr.reshape(t.shape)

    @property
    def total_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors.values())

    @property
    def total_params(self) -> int:
        return sum(t.n_elements for t in self.tensors.values())

    def close(self) -> None:
        self._mm.close()
        self._f.close()

    def __enter__(self) -> "SafeTensors":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
