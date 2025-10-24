import numpy as np
import unittest

def prepare_three_qubit_state(amplitudes):
    """
    Prepares a three-qubit quantum state vector from given amplitudes using tensor products (Stretch Goal).
    
    :param amplitudes: List or array of eight complex amplitudes [a0, ..., a7] for |000> to |111>.
    :return: Normalized NumPy array representing the state vector.
    """
    if len(amplitudes) != 8:
        raise ValueError("Exactly 8 amplitudes are required for a three-qubit state.")
    
    # Normalization step
    amps = np.array(amplitudes, dtype=complex)
    norm = np.linalg.norm(amps)
    if norm == 0:
        raise ValueError("Amplitudes cannot all be zero.")
    
    amps_normalized = amps / norm
    
    # Define single-qubit basis states
    zero = np.array([1.0 + 0j, 0.0 + 0j])
    one = np.array([0.0 + 0j, 1.0 + 0j])
    
    # Construct basis states and linear combination
    state = np.zeros(8, dtype=complex)
    basis_indices = [(0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)]
    for i, (q0, q1, q2) in enumerate(basis_indices):
        basis = np.kron([zero, one][q0], np.kron([zero, one][q1], [zero, one][q2]))
        state += amps_normalized[i] * basis
    
    return state

# Unit tests for three-qubit
class TestThreeQubitStatePreparation(unittest.TestCase):
    def test_three_qubit_normalization_enforced(self):
        amps = [1] * 8
        state = prepare_three_qubit_state(amps)
        self.assertAlmostEqual(np.sum(np.abs(state)**2), 1.0, places=10)
    
    def test_three_qubit_dimension(self):
        amps = [1] + [0] * 7
        state = prepare_three_qubit_state(amps)
        self.assertEqual(len(state), 8)
    
    def test_three_qubit_already_normalized(self):
        amps = [1/np.sqrt(2)] + [0]*6 + [1/np.sqrt(2)]
        state = prepare_three_qubit_state(amps)
        self.assertAlmostEqual(np.sum(np.abs(state)**2), 1.0, places=10)
        np.testing.assert_array_almost_equal(state, np.array(amps, dtype=complex))
    
    def test_three_qubit_complex_amplitudes(self):
        amps = [1, 1j, 1, 1j, 1, 1j, 1, 1j]
        state = prepare_three_qubit_state(amps)
        self.assertAlmostEqual(np.sum(np.abs(state)**2), 1.0, places=10)
    
    def test_three_qubit_zero_error(self):
        with self.assertRaises(ValueError):
            prepare_three_qubit_state([0] * 8)
    
    def test_three_qubit_length_error(self):
        with self.assertRaises(ValueError):
            prepare_three_qubit_state([1] * 7)

# To run the tests, uncomment the following:
# if __name__ == '__main__':
#     unittest.main()
