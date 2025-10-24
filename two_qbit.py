import numpy as np
import unittest

def prepare_two_qubit_state(amplitudes):
    """
    Prepares a two-qubit quantum state vector from given amplitudes using tensor products.
    
    :param amplitudes: List or array of four complex amplitudes [a0, a1, a2, a3] for |00>, |01>, |10>, |11>.
    :return: Normalized NumPy array representing the state vector.
    """
    if len(amplitudes) != 4:
        raise ValueError("Exactly 4 amplitudes are required for a two-qubit state.")
    
    # Normalization step: Compute the norm of the amplitudes
    amps = np.array(amplitudes, dtype=complex)
    norm = np.linalg.norm(amps)
    if norm == 0:
        raise ValueError("Amplitudes cannot all be zero.")
    
    # Normalize the amplitudes
    amps_normalized = amps / norm
    
    # Define single-qubit basis states
    zero = np.array([1.0 + 0j, 0.0 + 0j])
    one = np.array([0.0 + 0j, 1.0 + 0j])
    
    # Construct the state using tensor products: |ψ⟩ = a0|00⟩ + a1|01⟩ + a2|10⟩ + a3|11⟩
    state = (amps_normalized[0] * np.kron(zero, zero) +
             amps_normalized[1] * np.kron(zero, one) +
             amps_normalized[2] * np.kron(one, zero) +
             amps_normalized[3] * np.kron(one, one))
    
    return state

# Unit tests for two-qubit
class TestTwoQubitStatePreparation(unittest.TestCase):
    def test_two_qubit_normalization_enforced(self):
        # Non-normalized input
        amps = [1, 1, 1, 1]
        state = prepare_two_qubit_state(amps)
        self.assertAlmostEqual(np.sum(np.abs(state)**2), 1.0, places=10)
    
    def test_two_qubit_dimension(self):
        amps = [1, 0, 0, 0]
        state = prepare_two_qubit_state(amps)
        self.assertEqual(len(state), 4)
    
    def test_two_qubit_already_normalized(self):
        amps = [1/np.sqrt(2), 1/np.sqrt(2), 0, 0]
        state = prepare_two_qubit_state(amps)
        self.assertAlmostEqual(np.sum(np.abs(state)**2), 1.0, places=10)
        np.testing.assert_array_almost_equal(state, np.array([1/np.sqrt(2), 1/np.sqrt(2), 0, 0], dtype=complex))
    
    def test_two_qubit_complex_amplitudes(self):
        amps = [1, 1j, 1, 1j]
        state = prepare_two_qubit_state(amps)
        self.assertAlmostEqual(np.sum(np.abs(state)**2), 1.0, places=10)
    
    def test_two_qubit_zero_error(self):
        with self.assertRaises(ValueError):
            prepare_two_qubit_state([0, 0, 0, 0])
    
    def test_two_qubit_length_error(self):
        with self.assertRaises(ValueError):
            prepare_two_qubit_state([1, 2, 3])

# To run the tests, uncomment the following:
# if __name__ == '__main__':
#     unittest.main()
