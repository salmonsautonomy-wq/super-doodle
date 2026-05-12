import qutip as qt
import numpy as np
from typing import List, Optional, Tuple, Union


class QuantumSimulator:
    """
    A flexible quantum simulator built on QuTiP for open quantum systems.
    Supports state evolution, circuit simulation, and custom Hamiltonians.
    """

    def __init__(self, num_qubits: int = 2, use_qutip: bool = True):
        self.num_qubits = num_qubits
        self.use_qutip = use_qutip
        self.hilbert_dim = 2 ** num_qubits
        self.current_state = None
        self.history = []

    def initialize_state(self, state_type: str = "ground", alpha: float = 0.0) -> None:
        """Initialize the quantum state."""
        if state_type == "ground":
            self.current_state = qt.basis(self.hilbert_dim, 0)
        elif state_type == "coherent":
            # For multi-qubit, approximate coherent for first mode
            N = self.hilbert_dim
            self.current_state = qt.coherent(N, alpha)
        elif state_type == "bell":
            if self.num_qubits >= 2:
                self.current_state = (qt.bell_state('00') + qt.bell_state('11')).unit()
            else:
                self.current_state = qt.basis(self.hilbert_dim, 0)
        else:
            self.current_state = qt.basis(self.hilbert_dim, 0)
        self.history.append(self.current_state)

    def evolve(self, H: qt.Qobj, tlist: np.ndarray, c_ops: Optional[List[qt.Qobj]] = None,
               e_ops: Optional[List[qt.Qobj]] = None) -> qt.Result:
        """
        Evolve the system under a Hamiltonian with optional collapse operators.
        """
        if self.current_state is None:
            self.initialize_state()

        if c_ops is None:
            c_ops = []

        result = qt.mesolve(H, self.current_state, tlist, c_ops, e_ops)
        self.current_state = result.states[-1]
        self.history.extend(result.states)
        return result

    def apply_gate(self, gate: qt.Qobj, targets: List[int]) -> None:
        """Apply a unitary gate to specific qubits."""
        if self.current_state is None:
            self.initialize_state()

        # Build full operator
        full_op = self._tensor_gate(gate, targets)
        self.current_state = full_op * self.current_state
        self.history.append(self.current_state)

    def _tensor_gate(self, gate: qt.Qobj, targets: List[int]) -> qt.Qobj:
        """Create full Hilbert space operator for a gate on specific qubits."""
        ops = [qt.qeye(2) for _ in range(self.num_qubits)]
        for i, target in enumerate(targets):
            ops[target] = gate
        return qt.tensor(*ops)

    def expectation(self, op: qt.Qobj) -> float:
        """Compute expectation value of an operator."""
        if self.current_state is None:
            return 0.0
        return qt.expect(op, self.current_state)

    def measure(self, qubit: int, shots: int = 1000) -> dict:
        """Perform projective measurement on a qubit."""
        if self.current_state is None:
            self.initialize_state()

        # Projectors
        P0 = qt.tensor([qt.basis(2,0)*qt.basis(2,0).dag() if i==qubit else qt.qeye(2) for i in range(self.num_qubits)])
        P1 = qt.tensor([qt.basis(2,1)*qt.basis(2,1).dag() if i==qubit else qt.qeye(2) for i in range(self.num_qubits)])

        prob0 = qt.expect(P0, self.current_state)
        results = {'0': 0, '1': 0}

        for _ in range(shots):
            if np.random.random() < prob0:
                results['0'] += 1
            else:
                results['1'] += 1

        return {k: v/shots for k, v in results.items()}

    def get_state(self) -> qt.Qobj:
        """Return current quantum state."""
        return self.current_state

    def plot_bloch(self, qubit: int = 0):
        """Plot Bloch sphere for a qubit (if single or reduced)."""
        try:
            b = qt.Bloch()
            if self.num_qubits == 1:
                b.add_states(self.current_state)
            else:
                # Partial trace for multi-qubit
                rho = qt.ptrace(self.current_state, [qubit])
                b.add_states(rho)
            b.show()
        except Exception as e:
            print(f"Bloch plot error: {e}")


# Example usage
if __name__ == "__main__":
    print("=== QuantumSimulator Demo ===")
    sim = QuantumSimulator(num_qubits=2)
    sim.initialize_state("bell")

    # Example: Rabi oscillations or simple driven system
    sx = qt.sigmax()
    H = 2 * np.pi * 0.5 * qt.tensor(sx, qt.qeye(2))  # Simple Hamiltonian
    tlist = np.linspace(0, 10, 100)

    result = sim.evolve(H, tlist)
    print(f"Final expectation <σx>: {sim.expectation(qt.tensor(sx, qt.qeye(2))):.4f}")

    meas = sim.measure(0, shots=5000)
    print("Measurement probabilities:", meas)
    print("✅ QuantumSimulator ready for advanced simulations!")
