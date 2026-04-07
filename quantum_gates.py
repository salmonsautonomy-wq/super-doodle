class QuantumGate:
    def __init__(self, name):
        self.name = name

    def apply(self, state_vector):
        raise NotImplementedError("Apply method should be implemented by subclasses.")


class PauliX(QuantumGate):
    def __init__(self):
        super().__init__('Pauli-X')

    def apply(self, state_vector):
        # Perform Pauli-X operation on state_vector
        pass


class PauliY(QuantumGate):
    def __init__(self):
        super().__init__('Pauli-Y')

    def apply(self, state_vector):
        # Perform Pauli-Y operation on state_vector
        pass


class PauliZ(QuantumGate):
    def __init__(self):
        super().__init__('Pauli-Z')

    def apply(self, state_vector):
        # Perform Pauli-Z operation on state_vector
        pass


class Hadamard(QuantumGate):
    def __init__(self):
        super().__init__('Hadamard')

    def apply(self, state_vector):
        # Perform Hadamard operation on state_vector
        pass
