"""Guardrail against accidentally materializing a huge table.

Full collects (``.df``, ``.collect()``, ``.synapses``, ``.pairs``,
``PairUniverse.collect()``) first estimate the size of the result from a cheap
row count (``select(pl.len())``, which never builds the wide frame) and the
plan's schema. If the estimate is over the limit, they raise
:class:`CollectTooLargeError` *before* allocating anything.

The limit defaults to 25% of physical memory: polars' peak usage during joins
is often a few times the final frame, so a result that big already puts the
machine at risk. Override it with:

- ``trajan.set_max_collect_bytes(n)`` (``None`` disables the guard),
- the ``TRAJAN_MAX_COLLECT_BYTES`` environment variable (read at import;
  ``0`` or ``none`` disables),
- ``with trajan.allow_large_collect(): ...`` for a single block, or
- ``force=True`` on ``.collect()``.

The estimate is approximate: fixed-width dtypes are exact, while strings and
lists use a rough per-value guess. It is meant to catch an order-of-magnitude
mistake, not to budget memory precisely.
"""

from __future__ import annotations

import contextlib
import os
from typing import Iterator

import polars as pl

__all__ = [
    "CollectTooLargeError",
    "allow_large_collect",
    "estimate_bytes",
    "get_max_collect_bytes",
    "set_max_collect_bytes",
]

_FALLBACK_LIMIT = 4 * 1024**3  # used when physical memory can't be determined


class CollectTooLargeError(MemoryError):
    """Raised when a full collect is estimated to exceed the memory limit."""


def _default_limit() -> int | None:
    env = os.environ.get("TRAJAN_MAX_COLLECT_BYTES")
    if env is not None:
        if env.strip().lower() in {"", "0", "none", "off"}:
            return None
        return int(float(env))
    try:
        phys = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        return int(phys * 0.25)
    except (ValueError, OSError, AttributeError):
        return _FALLBACK_LIMIT


_max_bytes: int | None = _default_limit()
_allow_depth = 0


def get_max_collect_bytes() -> int | None:
    """Current collect limit in bytes (``None`` means the guard is off)."""
    return _max_bytes


def set_max_collect_bytes(n: int | None) -> None:
    """Set the collect limit in bytes. ``None`` disables the guard."""
    global _max_bytes
    if n is not None and n <= 0:
        raise ValueError("max collect bytes must be positive, or None to disable.")
    _max_bytes = n


@contextlib.contextmanager
def allow_large_collect() -> Iterator[None]:
    """Turn off the collect guard inside a ``with`` block."""
    global _allow_depth
    _allow_depth += 1
    try:
        yield
    finally:
        _allow_depth -= 1


_STRING_GUESS = 24  # bytes per value: offset + typical short payload
_LIST_GUESS_LEN = 4


def _dtype_bytes(dtype: pl.DataType) -> float:
    if isinstance(dtype, pl.Struct):
        return sum(_dtype_bytes(f.dtype) for f in dtype.fields)
    if isinstance(dtype, pl.Array):
        return dtype.size * _dtype_bytes(dtype.inner)
    if isinstance(dtype, pl.List):
        return 8 + _LIST_GUESS_LEN * _dtype_bytes(dtype.inner)
    if isinstance(dtype, (pl.String, pl.Binary)):
        return _STRING_GUESS
    if isinstance(dtype, (pl.Categorical, pl.Enum)):
        return 4
    if dtype == pl.Null:
        return 0
    if dtype == pl.Boolean:
        return 1
    if isinstance(dtype, pl.Decimal):
        return 16
    if dtype in (pl.Int8, pl.UInt8):
        return 1
    if dtype in (pl.Int16, pl.UInt16):
        return 2
    if dtype in (pl.Int32, pl.UInt32, pl.Float32, pl.Date):
        return 4
    if getattr(pl, "Int128", None) is not None and dtype == pl.Int128:
        return 16
    return 8  # 64-bit numerics, Datetime, Duration, Time, Object, unknown


def estimate_bytes(schema: pl.Schema, n_rows: int) -> int:
    """Rough in-memory size of ``n_rows`` rows with ``schema``."""
    per_row = sum(_dtype_bytes(dt) for dt in schema.dtypes())
    return int(per_row * n_rows)


def _fmt(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def check_collect(
    lf: pl.LazyFrame,
    what: str,
    *,
    n_rows: int | None = None,
    force: bool = False,
) -> None:
    """Raise :class:`CollectTooLargeError` if collecting ``lf`` looks too big.

    ``n_rows`` may be passed when the caller already has a cheap count (e.g. a
    cached base-table length); otherwise a ``select(pl.len())`` is run on the
    plan.
    """
    limit = _max_bytes
    if force or limit is None or _allow_depth > 0:
        return
    schema = lf.collect_schema()
    if n_rows is None:
        n_rows = lf.select(pl.len()).collect().item()
    est = estimate_bytes(schema, n_rows)
    if est <= limit:
        return
    # 1.5x headroom over the estimate, rounded up to a whole GiB (or MiB).
    unit = 1024**3 if est * 1.5 >= 1024**3 else 1024**2
    suggested = int(-(-est * 1.5 // unit) * unit)
    raise CollectTooLargeError(
        f"{what} would materialize ~{_fmt(est)} ({n_rows:,} rows x "
        f"{len(schema)} columns), over the trajan collect limit of "
        f"{_fmt(limit)}.\n"
        "Cheaper options:\n"
        "  .lazy / .select(...) / .group_by(...)   stay lazy, collect only the result\n"
        "  .collect(cols=[...])                     only the columns you need\n"
        "  .preview(n) / len(tbl) / .columns        inspect without materializing\n"
        "  .filter(...)                             shrink the table first\n"
        "To collect anyway:\n"
        "  .collect(force=True)                     this call only\n"
        "  with trajan.allow_large_collect(): ...   a block of code\n"
        f"  trajan.set_max_collect_bytes({suggested})"
        f"   raise the limit to ~{_fmt(suggested)} for this session\n"
        f"  export TRAJAN_MAX_COLLECT_BYTES={suggested}"
        "   raise it at import (0 disables)"
    )
