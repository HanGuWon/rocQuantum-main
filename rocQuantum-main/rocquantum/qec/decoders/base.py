"""Host-side decoder contracts, results, async wrapper, and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from threading import Lock
from typing import Callable, Dict, Iterator, List, Optional, Tuple, TypeVar, Union

import numpy as np

from .._gf2 import binary_matrix, readonly_copy, soft_probability_vector


def _strict_bool(value: object, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean.")
    return bool(value)


def _result_vector(values: object, name: str = "result") -> np.ndarray:
    try:
        raw = np.asarray(values, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a rank-1 vector of probabilities.") from exc
    if raw.ndim != 1:
        raise ValueError(f"{name} must be a rank-1 vector of probabilities.")
    return soft_probability_vector(raw, int(raw.size), name)


def _optional_results(
    values: Optional[Mapping[str, object]], name: str = "opt_results"
) -> Optional[Dict[str, object]]:
    if values is None:
        return None
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a string-keyed mapping or None.")
    result: Dict[str, object] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings.")
        result[key] = value
    return result


class DecoderResult:
    """One decoder outcome with validated soft bit-error probabilities."""

    def __init__(
        self,
        converged: object = False,
        result: Optional[object] = None,
        opt_results: Optional[Mapping[str, object]] = None,
    ) -> None:
        self.converged = converged
        self.result = np.empty(0, dtype=float) if result is None else result
        self.opt_results = opt_results

    @property
    def converged(self) -> bool:
        return self._converged

    @converged.setter
    def converged(self, value: object) -> None:
        self._converged = _strict_bool(value, "converged")

    @property
    def result(self) -> np.ndarray:
        return self._result.copy()

    @result.setter
    def result(self, value: object) -> None:
        self._result = readonly_copy(_result_vector(value))

    @property
    def opt_results(self) -> Optional[Dict[str, object]]:
        return None if self._opt_results is None else dict(self._opt_results)

    @opt_results.setter
    def opt_results(self, value: Optional[Mapping[str, object]]) -> None:
        self._opt_results = _optional_results(value)

    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int):
        values = (self.converged, self.result, self.opt_results)
        try:
            return values[index]
        except (IndexError, TypeError) as exc:
            raise IndexError("DecoderResult index must identify one of three fields.") from exc

    def __iter__(self):
        return iter((self.converged, self.result, self.opt_results))


class BatchDecoderResult(Sequence[DecoderResult]):
    """Vectorized decoder outcomes with one row per input syndrome."""

    def __init__(
        self,
        result: object,
        converged: object,
        opt_results: Optional[Sequence[Optional[Mapping[str, object]]]] = None,
    ) -> None:
        try:
            raw_converged = np.asarray(converged, dtype=object)
            raw_result = np.asarray(result, dtype=object)
        except (TypeError, ValueError) as exc:
            raise ValueError("Batch decoder results must be dense rank-1/rank-2 arrays.") from exc
        if raw_converged.ndim != 1 or raw_result.ndim != 2:
            raise ValueError("Batch converged/result values must have rank one and two.")
        if raw_result.shape[0] != raw_converged.size:
            raise ValueError("Batch result rows must match the converged vector length.")

        normalized_converged = np.empty(raw_converged.size, dtype=bool)
        for index, value in enumerate(raw_converged):
            normalized_converged[index] = _strict_bool(
                value, "batch converged entry"
            )
        normalized_result = np.empty(raw_result.shape, dtype=float)
        for row in range(raw_result.shape[0]):
            normalized_result[row] = soft_probability_vector(
                raw_result[row], raw_result.shape[1], "batch result row"
            )

        if opt_results is None:
            normalized_options = [None for _ in range(raw_converged.size)]
        else:
            if isinstance(opt_results, (str, bytes)) or not isinstance(
                opt_results, Sequence
            ):
                raise ValueError("opt_results must contain one mapping per batch row.")
            if len(opt_results) != raw_converged.size:
                raise ValueError("opt_results must contain one mapping per batch row.")
            normalized_options = [
                _optional_results(value, "batch opt_results entry")
                for value in opt_results
            ]

        self._converged = readonly_copy(normalized_converged)
        self._result = readonly_copy(normalized_result)
        self._opt_results = tuple(normalized_options)

    @classmethod
    def from_results(cls, results: Sequence[DecoderResult]) -> "BatchDecoderResult":
        if any(not isinstance(item, DecoderResult) for item in results):
            raise ValueError("results must contain only DecoderResult instances.")
        if not results:
            return cls(
                np.empty((0, 0), dtype=float),
                np.empty(0, dtype=bool),
                (),
            )
        width = results[0].result.size
        if any(item.result.size != width for item in results):
            raise ValueError("All decoder result rows must have the same width.")
        return cls(
            np.vstack([item.result for item in results]),
            [item.converged for item in results],
            [item.opt_results for item in results],
        )

    @property
    def converged(self) -> np.ndarray:
        return self._converged.copy()

    @property
    def result(self) -> np.ndarray:
        return self._result.copy()

    @property
    def opt_results(self):
        return [None if item is None else dict(item) for item in self._opt_results]

    @property
    def results(self) -> Tuple[DecoderResult, ...]:
        return tuple(self[index] for index in range(len(self)))

    def __len__(self) -> int:
        return int(self._result.shape[0])

    def __getitem__(self, index: Union[int, slice]):
        if isinstance(index, slice):
            return BatchDecoderResult(
                self._result[index],
                self._converged[index],
                self._opt_results[index],
            )
        return DecoderResult(
            bool(self._converged[index]),
            self._result[index],
            self._opt_results[index],
        )

    def __iter__(self) -> Iterator[DecoderResult]:
        for index in range(len(self)):
            yield self[index]


class AsyncDecoderResult:
    """Future-compatible wrapper returned by :meth:`Decoder.decode_async`."""

    def __init__(self, future: Future) -> None:
        if not isinstance(future, Future):
            raise ValueError("future must be a concurrent.futures.Future.")
        self._future = future

    def get(self, timeout: Optional[float] = None) -> DecoderResult:
        result = self._future.result(timeout=timeout)
        if not isinstance(result, DecoderResult):
            raise TypeError("Asynchronous decoder must return a DecoderResult.")
        return result

    def result(self, timeout: Optional[float] = None) -> DecoderResult:
        return self.get(timeout=timeout)

    def cancel(self) -> bool:
        return self._future.cancel()

    def cancelled(self) -> bool:
        return self._future.cancelled()

    def running(self) -> bool:
        return self._future.running()

    def done(self) -> bool:
        return self._future.done()

    def ready(self) -> bool:
        """Official-style non-blocking readiness check."""

        return self.done()

    def exception(self, timeout: Optional[float] = None):
        return self._future.exception(timeout=timeout)


_EXECUTOR: Optional[ThreadPoolExecutor] = None
_EXECUTOR_LOCK = Lock()


def _default_executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    with _EXECUTOR_LOCK:
        if _EXECUTOR is None:
            _EXECUTOR = ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="rocquantum-qec-decoder"
            )
        return _EXECUTOR


class Decoder(ABC):
    """Base class for strict dense-parity-check host decoders."""

    def __init__(self, H: object) -> None:
        normalized = binary_matrix(H, "H", allow_empty_rows=False)
        if normalized.shape[1] == 0:
            raise ValueError("H must contain at least one error-variable column.")
        self._H = readonly_copy(normalized)

    @property
    def H(self) -> np.ndarray:
        return self._H.copy()

    @H.setter
    def H(self, value: object) -> None:
        """Accept the same validated matrix assignment used by custom decoders.

        The parity matrix is immutable after construction because concrete
        decoders may precompute data structures from it.
        """

        normalized = binary_matrix(value, "H", allow_empty_rows=False)
        if normalized.shape[1] == 0:
            raise ValueError("H must contain at least one error-variable column.")
        if not np.array_equal(normalized, self._H):
            raise ValueError("A decoder parity-check matrix is immutable.")

    @property
    def syndrome_size(self) -> int:
        return int(self._H.shape[0])

    @property
    def block_size(self) -> int:
        return int(self._H.shape[1])

    def get_block_size(self) -> int:
        return self.block_size

    def get_syndrome_size(self) -> int:
        return self.syndrome_size

    def _normalize_syndrome(self, syndrome: object) -> Tuple[np.ndarray, np.ndarray]:
        soft = soft_probability_vector(
            syndrome, self.syndrome_size, "syndrome"
        )
        hard = (soft >= 0.5).astype(np.uint8)
        return soft, hard

    @abstractmethod
    def decode(self, syndrome: object) -> DecoderResult:
        """Decode one syndrome into per-variable soft error probabilities."""

    def decode_batch(self, syndromes: object) -> BatchDecoderResult:
        try:
            raw = np.asarray(syndromes, dtype=object)
        except (TypeError, ValueError) as exc:
            raise ValueError("syndromes must be a dense rank-2 probability matrix.") from exc
        if raw.ndim != 2 or raw.shape[1] != self.syndrome_size:
            raise ValueError(
                "syndromes must have one row per sample and one column per parity check."
            )
        results = [self.decode(raw[row]) for row in range(raw.shape[0])]
        if not results:
            return BatchDecoderResult(
                np.empty((0, 0), dtype=float),
                np.empty(0, dtype=bool),
                (),
            )
        return BatchDecoderResult.from_results(results)

    def decode_async(
        self, syndrome: object, executor: Optional[Executor] = None
    ) -> AsyncDecoderResult:
        soft, _ = self._normalize_syndrome(syndrome)
        selected_executor = _default_executor() if executor is None else executor
        if not isinstance(selected_executor, Executor):
            raise ValueError("executor must implement concurrent.futures.Executor.")
        return AsyncDecoderResult(selected_executor.submit(self.decode, soft.copy()))


DecoderFactory = Callable[..., Decoder]
_DECODER_REGISTRY: Dict[str, DecoderFactory] = {}
_FactoryT = TypeVar("_FactoryT", bound=DecoderFactory)


def _registry_name(name: object) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("decoder name must be a non-empty string.")
    return name.strip().lower()


def register_decoder(
    name: str, factory: Optional[_FactoryT] = None
) -> Callable[[_FactoryT], _FactoryT]:
    """Register a decoder factory directly or as a decorator."""

    normalized_name = _registry_name(name)

    def decorator(candidate: _FactoryT) -> _FactoryT:
        if not callable(candidate):
            raise ValueError("decoder factory must be callable.")
        if normalized_name in _DECODER_REGISTRY:
            raise ValueError(f"QEC decoder '{normalized_name}' is already registered.")
        _DECODER_REGISTRY[normalized_name] = candidate
        return candidate

    if factory is None:
        return decorator
    decorator(factory)
    return factory  # type: ignore[return-value]


def decoder(name: str) -> Callable[[_FactoryT], _FactoryT]:
    """Official-style decorator alias for :func:`register_decoder`."""

    return register_decoder(name)


def get_decoder(name: str, H: Optional[object] = None, **kwargs: object) -> Decoder:
    normalized_name = _registry_name(name)
    try:
        factory = _DECODER_REGISTRY[normalized_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown QEC decoder '{normalized_name}'. Available decoders: "
            f"{get_available_decoders()}."
        ) from exc
    instance = factory(**kwargs) if H is None else factory(H, **kwargs)
    if not isinstance(instance, Decoder):
        raise TypeError(
            f"Registered QEC decoder factory '{normalized_name}' must return a Decoder."
        )
    return instance


def get_available_decoders() -> List[str]:
    return sorted(_DECODER_REGISTRY)


__all__ = [
    "AsyncDecoderResult",
    "BatchDecoderResult",
    "Decoder",
    "DecoderResult",
    "decoder",
    "get_available_decoders",
    "get_decoder",
    "register_decoder",
]
