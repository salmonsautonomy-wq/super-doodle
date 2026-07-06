import numpy as np


class QuantumGate:
    """Base class for quantum gates. Stores a unitary matrix and applies it to state vectors."""

    def __init__(self, name: str, matrix):
        self.name = name
        self.matrix = np.array(matrix, dtype=complex)

    def apply(self, state_vector) -> np.ndarray:
        """Apply this gate's matrix directly to a state vector."""
        return self.matrix @ np.asarray(state_vector, dtype=complex)

    def __repr__(self):
        return f"{self.name}()"


class PauliX(QuantumGate):
    """Pauli-X (NOT) gate: flips |0⟩ ↔ |1⟩."""
    def __init__(self):
        super().__init__('Pauli-X', [[0, 1],
                                     [1, 0]])


class PauliY(QuantumGate):
    """Pauli-Y gate."""
    def __init__(self):
        super().__init__('Pauli-Y', [[0, -1j],
                                     [1j,  0]])


class PauliZ(QuantumGate):
    """Pauli-Z gate: flips the phase of |1⟩."""
    def __init__(self):
        super().__init__('Pauli-Z', [[1,  0],
                                     [0, -1]])


class Hadamard(QuantumGate):
    """Hadamard gate: creates equal superposition from |0⟩ or |1⟩."""
    def __init__(self):
        super().__init__('Hadamard', np.array([[1,  1],
                                               [1, -1]]) / np.sqrt(2))


class SGate(QuantumGate):
    """S (phase) gate: applies a π/2 phase to |1⟩."""
    def __init__(self):
        super().__init__('S', [[1, 0],
                               [0, 1j]])


class TGate(QuantumGate):
    """T (π/8) gate: applies a π/4 phase to |1⟩."""
    def __init__(self):
        super().__init__('T', [[1, 0],
                               [0, np.exp(1j * np.pi / 4)]])


class CNOT(QuantumGate):
    """CNOT (controlled-X) gate: flips the target qubit when the control qubit is |1⟩."""
    def __init__(self):
        super().__init__('CNOT', [[1, 0, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 0, 1],
                                  [0, 0, 1, 0]])
