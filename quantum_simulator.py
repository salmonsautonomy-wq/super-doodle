import numpy as np
from quantum_circuit import QuantumCircuit
from quantum_state import QuantumState

class QuantumSimulator:
    """Executes quantum circuits on a state vector simulator"""
    
    def __init__(self):
        """Initialize the quantum simulator"""
        self.state = None
    
    def run(self, circuit: QuantumCircuit, shots: int = 1) -> dict:
        """
        Execute a quantum circuit
        
        Args:
            circuit: QuantumCircuit to execute
            shots: Number of measurement shots
        
        Returns:
            Dictionary with measurement results
        """
        # Initialize state
        self.state = QuantumState(circuit.num_qubits)
        
        # Execute gates
        for op_type, gate, qubits in circuit.operations:
            self._apply_gate(gate, qubits)
        
        # Measure
        results = {}
        for _ in range(shots):
            # Get probabilities
            probabilities = np.abs(self.state.state_vector) ** 2
            outcome = np.random.choice(2 ** circuit.num_qubits, p=probabilities)
            basis_str = format(outcome, f'0{circuit.num_qubits}b')
            
            results[basis_str] = results.get(basis_str, 0) + 1
        
        return results
    
    def get_statevector(self, circuit: QuantumCircuit) -> np.ndarray:
        """Get the state vector after executing a circuit"""
        self.state = QuantumState(circuit.num_qubits)
        
        for op_type, gate, qubits in circuit.operations:
            self._apply_gate(gate, qubits)
        
        return self.state.get_state_vector()
    
    def _apply_gate(self, gate, qubits: list) -> None:
        """Apply a gate to the state"""
        gate_matrix = gate.get_matrix()
        num_qubits = self.state.num_qubits
        
        # Create full system operator using tensor products
        if len(qubits) == 1:
            qubit = qubits[0]
            # Single qubit gate
            full_operator = self._single_qubit_operator(gate_matrix, qubit, num_qubits)
        else:
            # Multi-qubit gate - for now support 2-qubit gates
            full_operator = self._multi_qubit_operator(gate_matrix, qubits, num_qubits)
        
        # Apply to state
        self.state.state_vector = full_operator @ self.state.state_vector
    
    def _single_qubit_operator(self, gate_matrix: np.ndarray, target: int, num_qubits: int) -> np.ndarray:
        """Create full system operator for single qubit gate"""
        # Build operator as tensor product of identity and gate
        operators = []
        for i in range(num_qubits):
            if i == target:
                operators.append(gate_matrix)
            else:
                operators.append(np.eye(2, dtype=complex))
        
        # Compute tensor product
        result = operators[0]
        for op in operators[1:]:
            result = np.kron(result, op)
        
        return result
    
    def _multi_qubit_operator(self, gate_matrix: np.ndarray, qubits: list, num_qubits: int) -> np.ndarray:
        """Create full system operator for multi-qubit gate"""
        # For 2-qubit gates
        if len(qubits) == 2:
            control, target = sorted(qubits)
            operators = []
            for i in range(num_qubits):
                if i < control:
                    operators.append(np.eye(2, dtype=complex))
                elif i == control:
                    # Build controlled operator
                    return self._apply_controlled_gate(gate_matrix, qubits, num_qubits)
                else:
                    operators.append(np.eye(2, dtype=complex))
        
        # Fallback
        return gate_matrix
    
    def _apply_controlled_gate(self, gate_matrix: np.ndarray, qubits: list, num_qubits: int) -> np.ndarray:
        """Apply a controlled gate"""
        control, target = qubits[0], qubits[1]
        
        # Create permutation matrix for controlled operation
        dim = 2 ** num_qubits
        full_operator = np.zeros((dim, dim), dtype=complex)
        
        for i in range(dim):
            # Check if control qubit is 1
            if (i >> (num_qubits - 1 - control)) & 1:
                # Apply gate to target qubit
                j = i
                if (i >> (num_qubits - 1 - target)) & 1:
                    # Target is 1, might flip
                    j = i ^ (1 << (num_qubits - 1 - target))
                
                full_operator[i, i] = gate_matrix[1, 1]
                if (j != i):
                    full_operator[i, j] = gate_matrix[1, 0]
                    full_operator[j, i] = gate_matrix[0, 1]
                    full_operator[j, j] = gate_matrix[0, 0]
            else:
                full_operator[i, i] = 1.0
        
        return full_operator
    
    def print_results(self, results: dict, shots: int) -> None:
        """Print measurement results"""
        print(f"\nMeasurement Results ({shots} shots):")
        for state, count in sorted(results.items()):
            percentage = 100 * count / shots
            print(f"  |{state}⟩: {count:4d} ({percentage:5.1f}%)")


# Example usage
if __name__ == "__main__":
    # Create a 2-qubit circuit
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cnot(0, 1)
    
    # Run simulator
    simulator = QuantumSimulator()
    results = simulator.run(circuit, shots=1000)
    simulator.print_results(results, shots=1000)
    
    # Print state vector
    state_vector = simulator.get_statevector(circuit)
    print("\nFinal state vector:", state_vector)