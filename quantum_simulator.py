import qutip as qt
import numpy as np
from typing import List, Optional

# QuTiP single-qubit gate objects used by run()
_QUTIP_SINGLE_GATES = {
    'h': qt.Qobj([[1,  1], [1, -1]], dims=[[2], [2]]) / np.sqrt(2),
    'x': qt.sigmax(),
    'y': qt.sigmay(),
    'z': qt.sigmaz(),
    's': qt.Qobj([[1, 0], [0, 1j]], dims=[[2], [2]]),
    't': qt.Qobj([[1, 0], [0, np.exp(1j * np.pi / 4)]], dims=[[2], [2]]),
}

_QUTIP_TWO_QUBIT_GATES = {
    'cnot': qt.Qobj(
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1],
         [0, 0, 1, 0]],
        dims=[[2, 2], [2, 2]]
    ),
}


class QuantumSimulator:
    """
    A flexible quantum simulator built on QuTiP for open quantum systems.
    Supports Hamiltonian evolution, circuit simulation, and custom measurements.
    """

    def __init__(self, num_qubits: int = 2, use_qutip: bool = True):
        self.num_qubits = num_qubits
        self.use_qutip = use_qutip
        self.hilbert_dim = 2 ** num_qubits
        self.current_state = None
        self.history = []

    # ── State initialisation ───────────────────────────────────────────────

    def initialize_state(self, state_type: str = "ground", alpha: float = 0.0) -> None:
        """
        Initialise the quantum state.

        state_type:
            "ground"   — |0⟩^⊗n  (default)
            "coherent" — coherent state (single-mode approximation, alpha sets amplitude)
            "bell"     — (|00⟩ + |11⟩)/√2  (only valid for num_qubits == 2)
        """
        if state_type == "ground":
            # Tensor product |0⟩^⊗n with proper QuTiP dims [[2]*n, [1]*n]
            self.current_state = qt.tensor([qt.basis(2, 0)] * self.num_qubits)

        elif state_type == "coherent":
            # Single-mode coherent state in the composite Hilbert space
            self.current_state = qt.coherent(self.hilbert_dim, alpha)

        elif state_type == "bell":
            if self.num_qubits != 2:
                raise ValueError(
                    f"Bell state requires exactly 2 qubits, "
                    f"but this simulator has {self.num_qubits}. "
                    "Use state_type='ground' for other qubit counts."
                )
            # Standard Bell state Φ+ = (|00⟩ + |11⟩) / √2
            self.current_state = qt.bell_state('00')

        else:
            raise ValueError(
                f"Unknown state_type '{state_type}'. "
                "Choose from: 'ground', 'coherent', 'bell'."
            )

        self.history.append(self.current_state)

    # ── Circuit execution ──────────────────────────────────────────────────

    def run(self, circuit, shots: int = 1000) -> dict:
        """
        Execute a QuantumCircuit and return measurement counts.

        Args:
            circuit: a QuantumCircuit instance
            shots:   number of measurement samples

        Returns:
            dict mapping basis-state labels (e.g. '00', '01') to counts
        """
        self.num_qubits = circuit.num_qubits
        self.hilbert_dim = 2 ** self.num_qubits
        self.history = []
        self.initialize_state("ground")

        for gate_name, qubits in circuit.circuit:
            gate_op = self._get_qutip_gate(gate_name)
            self.apply_gate(gate_op, qubits)

        return self._sample_all(shots)

    def _get_qutip_gate(self, gate_name: str) -> qt.Qobj:
        if gate_name in _QUTIP_SINGLE_GATES:
            return _QUTIP_SINGLE_GATES[gate_name]
        if gate_name in _QUTIP_TWO_QUBIT_GATES:
            return _QUTIP_TWO_QUBIT_GATES[gate_name]
        raise ValueError(
            f"Unknown gate '{gate_name}'. "
            f"Available: {sorted(list(_QUTIP_SINGLE_GATES) + list(_QUTIP_TWO_QUBIT_GATES))}"
        )

    def _sample_all(self, shots: int) -> dict:
        """Sample the full computational basis from the current state."""
        state_vec = self.current_state.full().flatten()
        probs = np.abs(state_vec) ** 2
        probs /= probs.sum()        # normalise away floating-point drift

        n = self.num_qubits
        dim = 2 ** n
        outcomes = np.random.choice(dim, size=shots, p=probs)
        counts = {}
        for outcome in outcomes:
            label = format(outcome, f'0{n}b')
            counts[label] = counts.get(label, 0) + 1
        return counts

    def print_results(self, results: dict, shots: int = 1000) -> None:
        """Print a histogram of measurement results."""
        print(f"\nMeasurement Results ({shots} shots):")
        for state_label in sorted(results):
            count = results[state_label]
            bar = '█' * int(40 * count / shots)
            print(f"  |{state_label}⟩: {count:5d}  {bar}  ({100 * count / shots:.1f}%)")

    # ── Gate application ───────────────────────────────────────────────────

    def apply_gate(self, gate: qt.Qobj, targets: List[int]) -> None:
        """Apply a unitary gate to specific qubits."""
        if self.current_state is None:
            self.initialize_state()

        full_op = self._tensor_gate(gate, targets)
        self.current_state = full_op * self.current_state
        self.history.append(self.current_state)

    def _tensor_gate(self, gate: qt.Qobj, targets: List[int]) -> qt.Qobj:
        """
        Build the full n-qubit unitary for a gate acting on specific qubits.

        Handles both single-qubit gates (tensored with identities on other
        qubits) and two-qubit gates (embedded via explicit index mapping).
        """
        gate_dim = gate.shape[0]
        gate_num_qubits = int(round(np.log2(gate_dim)))

        if gate_num_qubits == 1:
            if len(targets) != 1:
                raise ValueError(
                    f"Single-qubit gate requires 1 target, got {len(targets)}."
                )
            ops = [qt.qeye(2)] * self.num_qubits
            ops[targets[0]] = gate
            return qt.tensor(*ops)

        elif gate_num_qubits == 2:
            if len(targets) != 2:
                raise ValueError(
                    f"Two-qubit gate requires 2 targets, got {len(targets)}."
                )
            return self._embed_two_qubit_gate(gate, targets[0], targets[1])

        else:
            raise ValueError(
                f"Gates on {gate_num_qubits} qubits are not currently supported."
            )

    def _embed_two_qubit_gate(self, gate: qt.Qobj, q0: int, q1: int) -> qt.Qobj:
        """
        Embed a 4×4 two-qubit gate acting on qubits q0 (row-index 0) and
        q1 (row-index 1) into the full 2^n × 2^n Hilbert space.

        Convention: qubit 0 is the most-significant bit of the basis-state
        index (big-endian / QuTiP ordering).
        """
        n = self.num_qubits
        dim = 2 ** n
        gate_mat = gate.full()
        full_op = np.zeros((dim, dim), dtype=complex)

        for col in range(dim):
            col_bits = [(col >> (n - 1 - k)) & 1 for k in range(n)]
            for row in range(dim):
                row_bits = [(row >> (n - 1 - k)) & 1 for k in range(n)]
                # Non-gate qubits must be unchanged
                if any(row_bits[k] != col_bits[k]
                       for k in range(n) if k not in (q0, q1)):
                    continue
                col_sub = col_bits[q0] * 2 + col_bits[q1]
                row_sub = row_bits[q0] * 2 + row_bits[q1]
                full_op[row, col] = gate_mat[row_sub, col_sub]

        return qt.Qobj(full_op, dims=[[2] * n, [2] * n])

    # ── Hamiltonian evolution ──────────────────────────────────────────────

    def evolve(self, H: qt.Qobj, tlist: np.ndarray,
               c_ops: Optional[List[qt.Qobj]] = None,
               e_ops: Optional[List[qt.Qobj]] = None) -> qt.Result:
        """Evolve the system under a Hamiltonian with optional collapse operators."""
        if self.current_state is None:
            self.initialize_state()
        if c_ops is None:
            c_ops = []

        result = qt.mesolve(H, self.current_state, tlist,
                            c_ops=c_ops, e_ops=e_ops)
        self.current_state = result.states[-1]
        self.history.extend(result.states)
        return result

    # ── Observables & measurement ──────────────────────────────────────────

    def expectation(self, op: qt.Qobj) -> float:
        """Compute the expectation value of an operator."""
        if self.current_state is None:
            return 0.0
        return qt.expect(op, self.current_state)

    def measure(self, qubit: int, shots: int = 1000) -> dict:
        """
        Perform projective measurement on a single qubit.

        Returns a dict {'0': probability, '1': probability}.
        """
        if self.current_state is None:
            self.initialize_state()

        P0 = qt.tensor([
            qt.basis(2, 0) * qt.basis(2, 0).dag() if i == qubit else qt.qeye(2)
            for i in range(self.num_qubits)
        ])

        prob0 = float(np.real(qt.expect(P0, self.current_state)))
        prob0 = np.clip(prob0, 0.0, 1.0)

        counts = np.random.binomial(shots, prob0)
        return {'0': counts / shots, '1': (shots - counts) / shots}

    def get_state(self) -> qt.Qobj:
        """Return the current quantum state."""
        return self.current_state

    def plot_bloch(self, qubit: int = 0):
        """Plot the Bloch sphere for a single qubit (reduced density matrix for multi-qubit)."""
        try:
            b = qt.Bloch()
            if self.num_qubits == 1:
                b.add_states(self.current_state)
            else:
                rho = qt.ptrace(self.current_state, [qubit])
                b.add_states(rho)
            b.show()
        except Exception as e:
            print(f"Bloch plot error: {e}")


# ── Demo ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from quantum_circuit import QuantumCircuit

    print("=== QuantumSimulator Demo ===\n")

    # ── Demo 1: Bell state via circuit (matches README API) ────────────────
    circuit = QuantumCircuit(2)
    circuit.h(0)        # Hadamard on qubit 0 → superposition
    circuit.cnot(0, 1)  # CNOT: entangle qubits 0 and 1

    sim = QuantumSimulator()
    results = sim.run(circuit, shots=1000)
    print("Bell state circuit  (H on q0, CNOT q0→q1):")
    sim.print_results(results, shots=1000)

    # ── Demo 2: 3-qubit GHZ state ──────────────────────────────────────────
    ghz = QuantumCircuit(3)
    ghz.h(0)
    ghz.cnot(0, 1)
    ghz.cnot(0, 2)

    sim3 = QuantumSimulator()
    results3 = sim3.run(ghz, shots=1000)
    print("\nGHZ state  (H on q0, CNOT q0→q1, CNOT q0→q2):")
    sim3.print_results(results3, shots=1000)

    # ── Demo 3: Hamiltonian evolution ──────────────────────────────────────
    print("\nHamiltonian evolution  (driven 2-qubit Bell state):")
    sim2 = QuantumSimulator(num_qubits=2)
    sim2.initialize_state("bell")
    sx = qt.sigmax()
    H = 2 * np.pi * 0.5 * qt.tensor(sx, qt.qeye(2))
    tlist = np.linspace(0, 10, 100)
    sim2.evolve(H, tlist)
    print(f"  Final ⟨σx⟩ : {sim2.expectation(qt.tensor(sx, qt.qeye(2))):.4f}")
    meas = sim2.measure(0, shots=5000)
    print(f"  Qubit-0 measurement: {meas}")

    print("\n✅ QuantumSimulator ready for advanced simulations!")
