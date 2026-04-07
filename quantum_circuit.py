class QuantumCircuit:
    def __init__(self):
        self.circuit = []

    def add_gate(self, gate_name, qubits):
        self.circuit.append((gate_name, qubits))

    def __str__(self):
        return str(self.circuit)

    def run(self):
        # Placeholder for running the circuit
d        return "Circuit executed"