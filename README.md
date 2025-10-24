# Quantum State Preparation: Two-Qubit and Three-Qubit Implementations

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-yellow)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains Python implementations for manually preparing quantum states for two-qubit and three-qubit systems from a set of complex amplitudes. The code is written from scratch using fundamental quantum computing concepts, including **normalization**, **tensor products** (via NumPy's `np.kron`), and **matrix-vector representations**. No high-level quantum libraries (e.g., Qiskit, PennyLane) are used—everything is built using pure NumPy for numerical computations.

The routines construct the state vector directly as a linear combination of basis states, ensuring the output is a normalized NumPy array representing the quantum state. This aligns with quantum mechanics principles where the state |ψ⟩ must satisfy ∑|aᵢ|² = 1.

### Key Features
- **Two-Qubit Support**: Prepares |ψ⟩ = a₀|00⟩ + a₁|01⟩ + a₂|10⟩ + a₃|11⟩ from 4 complex amplitudes.
- **Three-Qubit Extension (Stretch Goal)**: Prepares |ψ⟩ = a₀|000⟩ + ... + a₇|111⟩ from 8 complex amplitudes.
- **Normalization**: Automatically normalizes unnormalized inputs; raises errors for invalid cases (e.g., all-zero amplitudes).
- **Input Validation**: Checks for correct amplitude count.
- **Unit Tests**: Comprehensive tests for normalization, dimension, complex amplitudes, and edge cases using `unittest`.
- **Efficiency**: O(1) time complexity for fixed qubit counts; lightweight tensor product operations.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/quantum-state-preparation.git
   cd quantum-state-preparation
   ```

2. Ensure Python 3.8+ and NumPy are installed:
   ```
   pip install numpy
   ```

No additional dependencies are required.

## Usage

### Two-Qubit State Preparation (`two_qbit.py`)

Import the function and provide a list of four complex amplitudes (as a list or NumPy array). The function returns a 4D complex NumPy array.

```python
import numpy as np
from two_qbit import prepare_two_qubit_state

# Example 1: Unnormalized Bell state
amplitudes = [1, 0, 0, 1]  # Will be normalized to [1/√2, 0, 0, 1/√2]
state = prepare_two_qubit_state(amplitudes)
print(state)  # [0.70710678+0.j 0.        +0.j 0.        +0.j 0.70710678+0.j]

# Example 2: Complex amplitudes
amplitudes = [1, 1j, 1, 1j]
state = prepare_two_qubit_state(amplitudes)
print(np.sum(np.abs(state)**2))  # 1.0 (normalized)
```

### Three-Qubit State Preparation (`three_qbit.py`)

Similar to the two-qubit version, but for eight amplitudes, returning an 8D array.

```python
from three_qbit import prepare_three_qubit_state

# Example: GHZ state (unnormalized)
amplitudes = [1, 0, 0, 0, 0, 0, 0, 1]
state = prepare_three_qubit_state(amplitudes)
print(state)  # [0.70710678+0.j 0.+0.j ... 0.70710678+0.j]

# Verify normalization
print(np.sum(np.abs(state)**2))  # 1.0
```

### Error Handling
- **Invalid Length**: Raises `ValueError` (e.g., fewer/more than 4 or 8 amplitudes).
- **Zero Norm**: Raises `ValueError` if all amplitudes are zero.

## Implementation Details

### Core Algorithm
1. **Validation**: Check input length.
2. **Normalization**:
   - Convert to complex NumPy array: `amps = np.array(amplitudes, dtype=complex)`.
   - Compute norm: `N = np.linalg.norm(amps)`.
   - If N == 0, raise error.
   - Scale: `amps /= N`.
3. **Basis States**:
   - Single-qubit: |0⟩ = [1+0j, 0+0j], |1⟩ = [0+0j, 1+0j].
   - Two-qubit: Use `np.kron` for |00⟩, |01⟩, |10⟩, |11⟩.
   - Three-qubit: Loop over binary indices (000 to 111) for triple Kronecker products.
4. **State Construction**: Linear combination `state = ∑ aᵢ * basis_i`.

This manual approach emphasizes tensor products as the foundation for multi-qubit states.

### File Structure
- `two_qbit.py`: Two-qubit routine + tests.
- `three_qbit.py`: Three-qubit routine (stretch goal) + tests.
- `README.md`: This file.

## Testing

Run tests for each file separately (uncomment the `if __name__ == '__main__':` block).

### Two-Qubit Tests (`two_qbit.py`)
- Normalization enforcement (unnormalized input).
- Output dimension (4).
- Already-normalized input.
- Complex amplitudes.
- Zero amplitudes error.
- Length error.

Example run:
```
python two_qbit.py
# Output: Ran 6 tests in 0.001s - OK
```

### Three-Qubit Tests (`three_qbit.py`)
- Normalization enforcement.
- Output dimension (8).
- Already-normalized input (e.g., GHZ state).
- Complex amplitudes.
- Zero amplitudes error.
- Length error.

Example run:
```
python three_qbit.py
# Output: Ran 6 tests in 0.001s - OK
```

All tests use `assertAlmostEqual` for floating-point precision and `np.testing.assert_array_almost_equal` for state vectors.

## Theoretical Notes
- **Hilbert Space**: Two qubits span ℂ⁴; three qubits span ℂ⁸.
- **Tensor Product**: Essential for composing multi-qubit systems (e.g., |ψ⟩ ⊗ |φ⟩ = Kronecker product).
- **Normalization**: Ensures Tr(ρ) = 1 for density matrix ρ = |ψ⟩⟨ψ|.

For circuit-based preparation (e.g., using R_y rotations and CNOT matrices), see the optional extension in the full project report.

## Contributing
Feel free to fork and submit pull requests for improvements, such as higher qubit counts or circuit simulations.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
Inspired by quantum computing fundamentals from Nielsen & Chuang's *Quantum Computation and Quantum Information*. Tested on Python 3.12 with NumPy 1.26.

---

*Repository created on October 24, 2025, for QOSF Cohort 11 Screening Test*
