"""Host-contract coverage for the CUDA-QX-inspired QEC surface."""

import itertools
import unittest

import numpy as np

from rocquantum.qec import (
    AsyncDecoderResult,
    BatchDecoderResult,
    BeliefPropagationDecoder,
    Code,
    CodeMetadata,
    DecoderResult,
    SingleErrorLUTDecoder,
    generate_random_bit_flips,
    get_available_codes,
    get_code,
    get_decoder,
    qec_capabilities,
    register_code,
    register_decoder,
    sample_code_capacity,
)
from rocquantum.qec._gf2 import gf2_rank
from rocq.operator import PauliOperator, QuantumOperator


class QECCodeRegistryTests(unittest.TestCase):
    def test_builtin_registry_and_odd_distance_repetition_metadata(self):
        self.assertEqual(get_available_codes(), ["repetition", "steane"])
        code = get_code("repetition", distance=5)
        self.assertEqual(code.metadata.name, "repetition")
        self.assertEqual(code.metadata.num_data_qubits, 5)
        self.assertEqual(code.metadata.num_logical_qubits, 1)
        self.assertEqual(code.metadata.distance, 5)
        self.assertEqual(code.metadata.num_ancilla_qubits, 4)
        self.assertEqual(code.H.shape, (4, 10))
        self.assertEqual(code.Hx.shape, (0, 5))
        self.assertEqual(code.Hz.shape, (4, 5))
        np.testing.assert_array_equal(
            code.Hz,
            np.asarray(
                (
                    (1, 1, 0, 0, 0),
                    (0, 1, 1, 0, 0),
                    (0, 0, 1, 1, 0),
                    (0, 0, 0, 1, 1),
                ),
                dtype=np.uint8,
            ),
        )
        self.assertEqual(code.logical_x.shape, (0, 5))
        self.assertEqual(code.get_observables_x().shape, (0, 5))
        np.testing.assert_array_equal(
            code.logical_z, np.asarray(((1, 0, 0, 0, 0),), np.uint8)
        )
        self.assertEqual(code.get_observables_z().shape, (1, 5))
        self.assertEqual(code.get_pauli_observables_matrix().shape, (1, 10))
        np.testing.assert_array_equal(
            code.get_pauli_observables_matrix()[0],
            np.asarray((0, 0, 0, 0, 0, 1, 0, 0, 0, 0), np.uint8),
        )

    def test_steane_commutation_rank_observables_and_metadata(self):
        code = get_code("steane")
        self.assertEqual(code.metadata.name, "steane")
        self.assertEqual(
            (
                code.metadata.num_data_qubits,
                code.metadata.num_logical_qubits,
                code.metadata.distance,
                code.metadata.num_ancilla_qubits,
            ),
            (7, 1, 3, 6),
        )
        self.assertEqual(code.H.dtype, np.uint8)
        self.assertEqual(code.H.shape, (6, 14))
        self.assertEqual(code.get_pauli_observables_matrix().shape, (2, 14))
        np.testing.assert_array_equal(
            code.get_pauli_observables_matrix()[0, :7], code.logical_x[0]
        )
        np.testing.assert_array_equal(
            code.get_pauli_observables_matrix()[1, 7:], code.logical_z[0]
        )
        self.assertEqual(gf2_rank(code.Hx), 3)
        self.assertEqual(gf2_rank(code.Hz), 3)
        self.assertFalse(np.any((code.Hx @ code.Hz.T) % 2))
        self.assertFalse(np.any((code.Hz @ code.logical_x.T) % 2))
        self.assertFalse(np.any((code.Hx @ code.logical_z.T) % 2))
        np.testing.assert_array_equal(
            (code.logical_x @ code.logical_z.T) % 2,
            np.ones((1, 1), dtype=np.uint8),
        )
        stabilizers = code.get_stabilizers()
        self.assertTrue(
            all(isinstance(item, QuantumOperator) for item in stabilizers)
        )
        self.assertTrue(all(isinstance(item, PauliOperator) for item in stabilizers))
        self.assertEqual(
            [item.pauli_string for item in stabilizers],
            [
                "X0 X1 X2 X3",
                "X1 X2 X4 X5",
                "X2 X3 X5 X6",
                "Z0 Z1 Z2 Z3",
                "Z1 Z2 Z4 Z5",
                "Z2 Z3 Z5 Z6",
            ],
        )
        self.assertEqual(
            {item.get_pauli_word() for item in stabilizers},
            {
                "ZZZZIII",
                "XXXXIII",
                "IXXIXXI",
                "IIXXIXX",
                "IZZIZZI",
                "IIZZIZZ",
            },
        )

    def test_asymmetric_observables_are_allowed_but_full_pairing_stays_strict(self):
        metadata = CodeMetadata(
            name="basis-specific",
            num_data_qubits=3,
            num_logical_qubits=1,
            distance=1,
            num_ancilla_qubits=0,
        )
        basis_specific = Code(
            metadata,
            Hx=np.zeros((0, 3), np.uint8),
            Hz=np.zeros((0, 3), np.uint8),
            logical_x=np.zeros((0, 3), np.uint8),
            logical_z=np.asarray(((1, 0, 0),), np.uint8),
        )
        self.assertEqual(basis_specific.get_observables_x().shape, (0, 3))
        self.assertEqual(basis_specific.get_observables_z().shape, (1, 3))
        self.assertEqual(
            basis_specific.get_pauli_observables_matrix().shape, (1, 6)
        )

        with self.assertRaisesRegex(
            ValueError, r"canonical GF\(2\) anticommutation"
        ):
            Code(
                metadata,
                Hx=np.zeros((0, 3), np.uint8),
                Hz=np.zeros((0, 3), np.uint8),
                logical_x=np.asarray(((1, 0, 0),), np.uint8),
                logical_z=np.asarray(((0, 1, 0),), np.uint8),
            )

        with self.assertRaisesRegex(ValueError, "more rows than encoded"):
            Code(
                metadata,
                Hx=np.zeros((0, 3), np.uint8),
                Hz=np.zeros((0, 3), np.uint8),
                logical_x=np.zeros((2, 3), np.uint8),
                logical_z=np.zeros((0, 3), np.uint8),
            )

    def test_duplicate_registry_entries_are_rejected_without_replacement(self):
        original_code = get_code("repetition")
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_code("repetition", lambda: original_code)
        original_decoder = get_decoder("single_error_lut", original_code.Hz)
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_decoder("single_error_lut", lambda H: original_decoder)
        self.assertEqual(get_code("repetition").distance, 3)
        self.assertIsInstance(
            get_decoder("single_error_lut", original_code.Hz),
            SingleErrorLUTDecoder,
        )

    def test_repetition_distance_validation_is_bool_safe(self):
        for value in (True, 1, 2, 4, 3.0, float("nan")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    get_code("repetition", distance=value)


class QECLUTDecoderTests(unittest.TestCase):
    def test_exhaustive_repetition_syndrome_classification(self):
        H = get_code("repetition", distance=5).Hz
        decoder = SingleErrorLUTDecoder(H)
        expected = {tuple([0] * H.shape[0]): None}
        expected.update(
            {tuple(int(bit) for bit in H[:, column]): column for column in range(5)}
        )
        for syndrome in itertools.product((0, 1), repeat=H.shape[0]):
            with self.subTest(syndrome=syndrome):
                decoded = decoder.decode(syndrome)
                self.assertEqual(decoded.converged, syndrome in expected)
                if syndrome in expected and expected[syndrome] is None:
                    np.testing.assert_array_equal(decoded.result, np.zeros(5))
                elif syndrome in expected:
                    self.assertEqual(int(np.argmax(decoded.result)), expected[syndrome])
                    self.assertEqual(float(decoded.result.sum()), 1.0)

    def test_all_steane_single_error_syndromes_are_decodable(self):
        H = get_code("steane").Hz
        decoder = get_decoder("single_error_lut", H)
        outcomes = {
            tuple(syndrome): decoder.decode(syndrome)
            for syndrome in itertools.product((0, 1), repeat=3)
        }
        self.assertEqual(len(outcomes), 8)
        self.assertTrue(all(result.converged for result in outcomes.values()))
        for column in range(7):
            result = outcomes[tuple(int(bit) for bit in H[:, column])]
            self.assertEqual(int(np.argmax(result.result)), column)

    def test_scalar_batch_and_async_results_agree(self):
        decoder = get_decoder("single_error_lut", get_code("repetition").Hz)
        self.assertEqual(decoder.get_block_size(), 3)
        self.assertEqual(decoder.get_syndrome_size(), 2)
        decoder.H = decoder.H
        syndromes = np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)), np.uint8)
        scalar = [decoder.decode(row) for row in syndromes]
        batch = decoder.decode_batch(syndromes)
        self.assertIsInstance(batch, BatchDecoderResult)
        self.assertEqual(len(batch), 4)
        np.testing.assert_array_equal(batch.converged, np.ones(4, dtype=bool))
        np.testing.assert_array_equal(
            batch.result, np.vstack([result.result for result in scalar])
        )
        pending = decoder.decode_async(syndromes[2])
        self.assertIsInstance(pending, AsyncDecoderResult)
        async_result = pending.get(timeout=2.0)
        self.assertTrue(pending.ready())
        self.assertTrue(async_result.converged)
        np.testing.assert_array_equal(async_result.result, scalar[2].result)
        sliced = batch[1:3]
        self.assertIsInstance(sliced, BatchDecoderResult)
        np.testing.assert_array_equal(sliced.result, batch.result[1:3])
        self.assertIsNone(scalar[0].opt_results)
        self.assertEqual(len(scalar[0]), 3)
        self.assertIs(scalar[0][0], True)

    def test_pcm_syndrome_and_result_validation_reject_bool_and_nan(self):
        invalid_matrices = (
            [[True, 0], [0, 1]],
            [[0.0, 1.0]],
            [[0, float("nan")]],
            [[0, 2]],
            [0, 1],
        )
        for matrix in invalid_matrices:
            with self.subTest(matrix=matrix):
                with self.assertRaises(ValueError):
                    SingleErrorLUTDecoder(matrix)

        decoder = SingleErrorLUTDecoder([[1, 1, 0], [0, 1, 1]])
        for syndrome in ([True, 0], [float("nan"), 0], [-0.1, 0], [1.1, 0], [0]):
            with self.subTest(syndrome=syndrome):
                with self.assertRaises(ValueError):
                    decoder.decode(syndrome)
        with self.assertRaises(ValueError):
            DecoderResult(True, [float("nan")])
        with self.assertRaises(ValueError):
            DecoderResult(1, [0.0])
        with self.assertRaises(ValueError):
            BatchDecoderResult([[True]], [True])


class QECCapacityTests(unittest.TestCase):
    def test_random_bit_flips_are_seeded_and_cover_probability_boundaries(self):
        first = generate_random_bit_flips(64, 0.25, seed=42)
        second = generate_random_bit_flips(64, 0.25, seed=42)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.dtype, np.uint8)
        np.testing.assert_array_equal(
            generate_random_bit_flips(7, 0.0, seed=1), np.zeros(7, np.uint8)
        )
        np.testing.assert_array_equal(
            generate_random_bit_flips(7, 1.0, seed=1), np.ones(7, np.uint8)
        )

    def test_capacity_sampling_preserves_exact_gf2_invariant(self):
        code = get_code("steane")
        first = sample_code_capacity(code, 128, 0.2, seed=8)
        second = sample_code_capacity(code, 128, 0.2, seed=8)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        syndromes, errors = first
        self.assertEqual(syndromes.shape, (128, 6))
        self.assertEqual(errors.shape, (128, 14))
        self.assertEqual(syndromes.dtype, np.uint8)
        self.assertEqual(errors.dtype, np.uint8)
        np.testing.assert_array_equal(syndromes, (errors @ code.H.T) % 2)

        for probability in (0.0, 1.0):
            syndromes, errors = sample_code_capacity(
                code.Hz, 5, probability, seed=4
            )
            np.testing.assert_array_equal(syndromes, (errors @ code.Hz.T) % 2)
            np.testing.assert_array_equal(
                errors, np.full((5, 7), int(probability), dtype=np.uint8)
            )

    def test_capacity_validation_is_bool_and_nan_safe(self):
        H = [[1, 1, 0], [0, 1, 1]]
        for args in (
            (H, True, 0.1, None),
            (H, 1, True, None),
            (H, 1, float("nan"), None),
            ([[True, 0]], 1, 0.1, None),
            (H, 1, 0.1, True),
        ):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    sample_code_capacity(*args)


class QECBeliefPropagationTests(unittest.TestCase):
    def test_repetition_zero_and_single_error_syndromes_converge(self):
        H = get_code("repetition", distance=5).Hz
        decoder = BeliefPropagationDecoder(
            H, error_probability=0.05, max_iterations=25
        )
        zero = decoder.decode(np.zeros(H.shape[0], np.uint8))
        self.assertTrue(zero.converged)
        self.assertTrue(np.all(zero.result < 0.5))
        for error_index in range(H.shape[1]):
            syndrome = H[:, error_index]
            with self.subTest(error_index=error_index):
                result = decoder.decode(syndrome)
                self.assertTrue(result.converged)
                self.assertEqual(int(np.argmax(result.result)), error_index)

    def test_belief_propagation_prior_validation(self):
        H = get_code("repetition").Hz
        for prior in (True, 0.0, 1.0, float("nan"), [0.1, 0.1]):
            with self.subTest(prior=prior):
                with self.assertRaises(ValueError):
                    BeliefPropagationDecoder(H, error_probability=prior)
        with self.assertRaises(ValueError):
            BeliefPropagationDecoder(H, max_iterations=True)

    def test_capabilities_report_feature_level_evidence_without_parity_claim(self):
        capabilities = qec_capabilities()
        evidence = capabilities["feature_evidence"]
        self.assertEqual(evidence["native_rocm_execution"]["status"], "not_hardware_tested")
        self.assertEqual(
            evidence["tensor_network_and_realtime_qec"]["status"], "unsupported"
        )
        self.assertEqual(capabilities["available_codes"], ["repetition", "steane"])
        self.assertEqual(
            capabilities["available_host_decoders"],
            ["belief_propagation", "single_error_lut"],
        )
        self.assertEqual(
            capabilities["execution_scope"]["decoder_library_scope"],
            "host_single_error_lut_and_dense_sum_product_belief_propagation",
        )


if __name__ == "__main__":
    unittest.main()
