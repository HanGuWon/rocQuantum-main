"""Compatibility result types for the canonical :mod:`rocq` runtime."""

from __future__ import annotations

import math
from concurrent.futures import Future, InvalidStateError
from numbers import Integral, Number
from typing import Mapping, Optional


class SampleResult(dict):
    """Sampling counts with full ``dict`` compatibility."""

    def __init__(self, counts: Optional[Mapping[str, int]] = None, **kwargs):
        super().__init__()
        raw_counts = {} if counts is None else dict(counts)
        raw_counts.update(kwargs)
        for bitstring, count in raw_counts.items():
            if not isinstance(bitstring, str):
                raise TypeError("SampleResult bitstrings must be strings.")
            if isinstance(count, bool) or not isinstance(count, Integral):
                raise TypeError("SampleResult counts must be non-negative integers.")
            normalized_count = int(count)
            if normalized_count < 0:
                raise ValueError("SampleResult counts must be non-negative integers.")
            self[bitstring] = normalized_count

    @property
    def total_shots(self) -> int:
        """Return the total number of recorded shots."""

        return sum(self.values())

    def most_probable(self) -> str:
        """Return the most frequent bitstring, breaking ties lexicographically."""

        if not self:
            raise ValueError("Cannot select a most-probable value from an empty result.")
        return min(self, key=lambda key: (-self[key], key))

    def probability(self, bitstring: str) -> float:
        """Return the empirical probability for ``bitstring``."""

        total = self.total_shots
        if total == 0:
            return 0.0
        return self.get(bitstring, 0) / total


class ObserveResult(float):
    """A real expectation value with ``float`` compatibility."""

    def __new__(cls, value):
        if isinstance(value, bool) or not isinstance(value, Number):
            raise TypeError("ObserveResult requires a finite real number.")
        scalar = complex(value)
        if not math.isclose(scalar.imag, 0.0, abs_tol=1e-12):
            raise ValueError("ObserveResult requires a real expectation value.")
        normalized = float(scalar.real)
        if not math.isfinite(normalized):
            raise ValueError("ObserveResult requires a finite real number.")
        return float.__new__(cls, normalized)

    def expectation(self) -> float:
        """Return the expectation value as a plain float."""

        return float(self)


class AsyncResult(Future):
    """A :class:`Future` wrapper that also provides CUDA-Q-style ``get``."""

    def __init__(self, source: Future):
        if not isinstance(source, Future):
            raise TypeError("AsyncResult requires a concurrent.futures.Future.")
        super().__init__()
        self._source = source
        source.add_done_callback(self._complete_from_source)

    def _complete_from_source(self, source: Future) -> None:
        if self.done():
            return
        if source.cancelled():
            super().cancel()
            return
        try:
            value = source.result()
        except BaseException as exc:
            try:
                self.set_exception(exc)
            except InvalidStateError:
                pass
            return
        try:
            self.set_result(value)
        except InvalidStateError:
            pass

    def cancel(self) -> bool:
        if self.done():
            return False
        if not self._source.cancel():
            return False
        return super().cancel()

    def running(self) -> bool:
        return not self.done() and self._source.running()

    def get(self, timeout: Optional[float] = None):
        """Wait for and return the asynchronous result."""

        return self.result(timeout=timeout)


__all__ = ["SampleResult", "ObserveResult", "AsyncResult"]
