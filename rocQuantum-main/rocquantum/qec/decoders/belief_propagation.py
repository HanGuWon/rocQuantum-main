"""Pure-NumPy sum-product belief-propagation decoder."""

from __future__ import annotations

import math

import numpy as np

from .._gf2 import positive_integer, probability, readonly_copy, soft_probability_vector
from .base import Decoder, DecoderResult, register_decoder
from .lut import SingleErrorLUTDecoder


@register_decoder("belief_propagation")
class BeliefPropagationDecoder(Decoder):
    """Host sum-product decoder for a dense binary parity-check matrix."""

    def __init__(
        self,
        H: object,
        error_probability: object = 0.05,
        max_iterations: int = 50,
    ) -> None:
        super().__init__(H)
        self.max_iterations = positive_integer(max_iterations, "max_iterations")
        try:
            raw_prior = np.asarray(error_probability, dtype=object)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "error_probability must be a scalar or one value per error variable."
            ) from exc
        if raw_prior.ndim == 0:
            scalar = probability(raw_prior.item(), "error_probability")
            prior = np.full(self.block_size, scalar, dtype=float)
        elif raw_prior.ndim == 1 and raw_prior.size == self.block_size:
            prior = soft_probability_vector(
                raw_prior, self.block_size, "error_probability"
            )
        else:
            raise ValueError(
                "error_probability must be a scalar or one value per error variable."
            )
        if np.any(prior <= 0.0) or np.any(prior >= 1.0):
            raise ValueError(
                "Belief-propagation error probabilities must lie strictly between 0 and 1."
            )
        self._error_probability = readonly_copy(prior)

    @property
    def error_probability(self) -> np.ndarray:
        return self._error_probability.copy()

    @staticmethod
    def _error_probabilities(llr: np.ndarray) -> np.ndarray:
        clipped = np.clip(llr, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(clipped))

    def decode(self, syndrome: object) -> DecoderResult:
        _, hard_syndrome = self._normalize_syndrome(syndrome)
        prior_llr = np.log((1.0 - self._error_probability) / self._error_probability)
        variable_to_check = np.zeros_like(self._H, dtype=float)
        for check, variable in zip(*np.nonzero(self._H)):
            variable_to_check[check, variable] = prior_llr[variable]

        posterior = prior_llr.copy()
        for iteration in range(1, self.max_iterations + 1):
            check_to_variable = np.zeros_like(variable_to_check)
            for check in range(self.syndrome_size):
                neighbors = np.flatnonzero(self._H[check])
                syndrome_sign = -1.0 if hard_syndrome[check] else 1.0
                for variable in neighbors:
                    product = syndrome_sign
                    for other in neighbors:
                        if other != variable:
                            product *= math.tanh(
                                variable_to_check[check, other] / 2.0
                            )
                    product = min(max(product, -1.0 + 1e-12), 1.0 - 1e-12)
                    check_to_variable[check, variable] = 2.0 * math.atanh(product)

            posterior = prior_llr + check_to_variable.sum(axis=0)
            decision = (posterior < 0.0).astype(np.uint8)
            decoded_syndrome = (self._H @ decision) % 2
            if np.array_equal(decoded_syndrome, hard_syndrome):
                return DecoderResult(
                    True,
                    self._error_probabilities(posterior),
                )

            for check, variable in zip(*np.nonzero(self._H)):
                variable_to_check[check, variable] = (
                    prior_llr[variable]
                    + check_to_variable[:, variable].sum()
                    - check_to_variable[check, variable]
                )

        # A unique single-error syndrome is still exactly decidable when a
        # short loopy graph prevents sum-product convergence.
        fallback = SingleErrorLUTDecoder(self._H).decode(hard_syndrome)
        if fallback.converged:
            return DecoderResult(
                True,
                fallback.result,
            )
        return DecoderResult(
            False,
            self._error_probabilities(posterior),
        )


__all__ = ["BeliefPropagationDecoder"]
