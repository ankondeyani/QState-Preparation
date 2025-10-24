import numpy as np
import unittest
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

def prepare_three_qubit_state(amplitudes):
    """
    Prepares a three-qubit quantum state vector from given amplitudes using tensor products.
    
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

def ry_matrix(theta):
    """R_y rotation matrix."""
    return np.array([[np.cos(theta/2), -np.sin(theta/2)],
                     [np.sin(theta/2), np.cos(theta/2)]], dtype=complex)

def rz_matrix(phi):
    """R_z rotation matrix."""
    return np.array([[np.exp(-1j*phi/2), 0],
                     [0, np.exp(1j*phi/2)]], dtype=complex)

def cnot_matrix(control, target, n_qubits=3):
    """CNOT matrix for n-qubit system."""
    dim = 2**n_qubits
    CNOT = np.eye(dim, dtype=complex)
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    P0 = np.array([[1, 0], [0, 0]], dtype=complex)
    P1 = np.array([[0, 0], [0, 1]], dtype=complex)
    if control == 0 and target == 1:
        CNOT = np.kron(P0, np.kron(I, I)) + np.kron(P1, np.kron(X, I))
    elif control == 0 and target == 2:
        CNOT = np.kron(P0, np.kron(I, I)) + np.kron(P1, np.kron(I, X))
    elif control == 1 and target == 2:
        CNOT = np.kron(I, np.kron(P0, I)) + np.kron(I, np.kron(P1, X))
    return CNOT

def prepare_three_qubit_state_circuit(amplitudes):
    """
    Prepares a three-qubit quantum state via a simplified circuit from |000> (for illustration).
    
    :param amplitudes: List of eight complex amplitudes [a0, ..., a7].
    :return: Normalized 8D state vector (simplified; full decomposition omitted for brevity).
    """
    if len(amplitudes) != 8:
        raise ValueError("Exactly 8 amplitudes required for three-qubit state.")
    
    amps = np.array(amplitudes, dtype=complex)
    norm = np.linalg.norm(amps)
    if norm == 0:
        raise ValueError("Amplitudes cannot all be zero.")
    amps = amps / norm
    
    # Simplified parameterization (for demo; full would use hierarchical angles)
    p0 = np.sqrt(np.abs(amps[0])**2 + np.abs(amps[1])**2 + np.abs(amps[2])**2 + np.abs(amps[3])**2)
    theta1 = 2 * np.arccos(p0) if p0 > 0 else np.pi
    
    # Initial state |000>
    state = np.array([1.0] + [0.0]*7, dtype=complex)
    
    I = np.eye(2, dtype=complex)
    
    # Apply R_y(theta1) on q0
    state = np.kron(ry_matrix(theta1), np.kron(I, I)) @ state
    
    # Additional gates (placeholder for full circuit)
    state = cnot_matrix(0, 1) @ state
    state = cnot_matrix(1, 2) @ state
    
    return state / np.linalg.norm(state)  # Renormalize

def compute_probabilities(state):
    """
    Computes measurement probabilities |a_i|^2 for each basis state.
    
    :param state: 8D complex NumPy array (three-qubit state).
    :return: Dict with basis states as keys and probabilities as values.
    """
    probs = {f'|{bin(i)[2:].zfill(3)}>': np.abs(state[i])**2 for i in range(8)}
    return probs

def fidelity(state1, state2):
    """
    Computes fidelity F = |<psi|phi>|^2 between two states.
    
    :param state1, state2: Complex NumPy arrays of same dimension.
    :return: Fidelity value (float).
    """
    if len(state1) != len(state2):
        raise ValueError("States must have the same dimension.")
    overlap = np.abs(np.dot(state1.conj(), state2))**2
    return overlap

def plot_state_probabilities(state, title='Three-Qubit State Probabilities'):
    """
    Plots bar chart of measurement probabilities.
    
    :param state: 8D complex NumPy array.
    :param title: Plot title.
    """
    probs = [np.abs(s)**2 for s in state]
    labels = [f'|{bin(i)[2:].zfill(3)}>' for i in range(8)]
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, probs)
    plt.xlabel('Basis States')
    plt.ylabel('Probability')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('state_probs.png')
    plt.show()
    print("Plot saved as 'state_probs.png'")

# Unit tests
class TestThreeQubitExtended(unittest.TestCase):
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
    
    def test_three_qubit_w_state(self):
        # W state: (1/√3)(|001> + |010> + |100>)
        amps = [0, 1/np.sqrt(3), 1/np.sqrt(3), 0, 1/np.sqrt(3), 0, 0, 0]
        state = prepare_three_qubit_state(amps)
        expected_probs = {k: 1/3 for k in ['|001>', '|010>', '|100>']}
        probs = compute_probabilities(state)
        for k in expected_probs:
            self.assertAlmostEqual(probs[k], expected_probs[k], places=10)
    
    def test_three_qubit_fidelity(self):
        amps1 = [1] + [0]*7
        amps2 = [0, 0, 0, 0, 0, 0, 0, 1]
        state1 = prepare_three_qubit_state(amps1)
        state2 = prepare_three_qubit_state(amps2)
        fid = fidelity(state1, state2)
        self.assertAlmostEqual(fid, 0.0, places=10)
    
    def test_three_qubit_zero_error(self):
        with self.assertRaises(ValueError):
            prepare_three_qubit_state([0] * 8)
    
    def test_three_qubit_length_error(self):
        with self.assertRaises(ValueError):
            prepare_three_qubit_state([1] * 7)
    
    def test_circuit_vs_direct(self):
        amps = [1/np.sqrt(2)] + [0]*6 + [1/np.sqrt(2)]
        state_direct = prepare_three_qubit_state(amps)
        state_circuit = prepare_three_qubit_state_circuit(amps)
        fid = fidelity(state_direct, state_circuit)
        self.assertGreater(fid, 0.5)  # Approximate match for simplified circuit

# To run the tests, uncomment the following:
# if __name__ == '__main__':
#     unittest.main()
    # Example usage:
    # amps = [1/np.sqrt(2)] + [0]*6 + [1/np.sqrt(2)]
    # state = prepare_three_qubit_state(amps)
    # probs = compute_probabilities(state)
    # print(probs)
    # plot_state_probabilities(state)
