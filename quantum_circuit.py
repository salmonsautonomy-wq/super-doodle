import numpy as np
from quantum_gates import PauliX, PauliY, PauliZ, Hadamard, SGate, TGate, CNOT

# Map gate names to their classes
GATE_MAP = {
    'x':    PauliX,
    'y':    PauliY,
    'z':    PauliZ,
    'h':    Hadamard,
    's':    SGate,
    't':    TGate,
    'cnot': CNOT,
}


class QuantumCircuit:
    """
    Builds and executes a sequence of quantum gates on an n-qubit register.

    Usage:
        circuit = QuantumCircuit(2)
        circuit.h(0)          # Hadamard on qubit 0
        circuit.cnot(0, 1)    # CNOT: control=0, target=1
        state = circuit.run() # returns final numpy state vector
    """

    def __init__(self, num_qubits: int):
        if num_qubits < 1:
            raise ValueError("num_qubits must be at least 1.")
        self.num_qubits = num_qubits
        self.circuit = []   # list of (gate_name, [qubit_indices])

    # ── Convenience gate methods ───────────────────────────────────────────

    def h(self, qubit: int):
        """Apply Hadamard to qubit."""
        self._add('h', [qubit])

    def x(self, qubit: int):
        """Apply Pauli-X (NOT) to qubit."""
        self._add('x', [qubit])

    def y(self, qubit: int):
        """Apply Pauli-Y to qubit."""
        self._add('y', [qubit])

    def z(self, qubit: int):
        """Apply Pauli-Z to qubit."""
        self._add('z', [qubit])

    def s(self, qubit: int):
        """Apply S (phase) gate to qubit."""
        self._add('s', [qubit])

    def t(self, qubit: int):
        """Apply T gate to qubit."""
        self._add('t', [qubit])

    def cnot(self, control: int, target: int):
        """Apply CNOT with given control and target qubits."""
        self._add('cnot', [control, target])

    def add_gate(self, gate_name: str, qubits):
        """Add a gate by name (string) and qubit list. Kept for backward compatibility."""
        self._add(gate_name.lower(), list(qubits))

    # ── Execution ──────────────────────────────────────────────────────────

    def run(self) -> np.ndarray:
        """
        Execute all gates on |0…0⟩ and return the final state vector as a
        numpy array of length 2**num_qubits.
        """
        dim = 2 ** self.num_qubits
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0          # start in |0…0⟩

        for gate_name, qubits in self.circuit:
            if gate_name not in GATE_MAP:
                raise ValueError(
                    f"Unknown gate '{gate_name}'. Available: {sorted(GATE_MAP)}"
                )
            op = self._build_operator(gate_name, qubits)
            state = op @ state

        return state

    # ── Internal helpers ───────────────────────────────────────────────────

    def _add(self, gate_name: str, qubits: list):
        n = self.num_qubits
        for q in qubits:
            if not (0 <= q < n):
                raise ValueError(
                    f"Qubit index {q} out of range for {n}-qubit circuit."
                )
        self._validate_gate(gate_name, qubits)
        self.circuit.append((gate_name, qubits))

    def _validate_gate(self, gate_name: str, qubits: list):
        two_qubit = {'cnot'}
        if gate_name in two_qubit and len(qubits) != 2:
            raise ValueError(f"'{gate_name}' requires exactly 2 qubits.")
        if gate_name not in two_qubit and len(qubits) != 1:
            raise ValueError(f"'{gate_name}' requires exactly 1 qubit.")
        if gate_name == 'cnot' and qubits[0] == qubits[1]:
            raise ValueError("CNOT control and target must be different qubits.")

    def _build_operator(self, gate_name: str, qubits: list) -> np.ndarray:
        """
        Build the full 2^n × 2^n unitary matrix for a gate acting on the
        specified qubits within the n-qubit Hilbert space.
        """
        n = self.num_qubits
        gate = GATE_MAP[gate_name]()

        if gate_name == 'cnot':
            return self._embed_cnot(qubits[0], qubits[1])

        # Single-qubit gate: tensor product with identities on other qubits
        qubit = qubits[0]
        ops = [np.eye(2, dtype=complex)] * n
        ops[qubit] = gate.matrix
        result = ops[0]
        for m in ops[1:]:
            result = np.kron(result, m)
        return result

    def _embed_cnot(self, control: int, target: int) -> np.ndarray:
        """
        Build the 2^n × 2^n CNOT unitary for arbitrary control and target
        qubit positions within the n-qubit register.

        Convention: qubit 0 is the most-significant bit of the basis-state
        index (same as QuTiP / Qiskit big-endian ordering).
        """
        n = self.num_qubits
        dim = 2 ** n
        op = np.eye(dim, dtype=complex)

        for col in range(dim):
            # bit k of col: (col >> (n-1-k)) & 1
            ctrl_bit = (col >> (n - 1 - control)) & 1
            if ctrl_bit == 1:
                # Flip target bit
                flip_mask = 1 << (n - 1 - target)
                row = col ^ flip_mask
                # Swap columns col: set op[col,col]=0, op[row,col]=1
                op[col, col] = 0
                op[row, col] = 1

        return op

    def __str__(self):
        lines = [f"QuantumCircuit({self.num_qubits} qubits)"]
        for gate_name, qubits in self.circuit:
            lines.append(f"  {gate_name.upper():<6} {qubits}")
        return "\n".join(lines)
