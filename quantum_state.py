class QuantumState:
    def __init__(self, state_vector):
        self.state_vector = state_vector

    def normalize(self):
        norm = sum(abs(amplitude)**2 for amplitude in self.state_vector) ** 0.5
        self.state_vector = [amplitude / norm for amplitude in self.state_vector]

    def __str__(self):
        return f'QuantumState({self.state_vector})'

    def measure(self):
        import random
        probabilities = [abs(amplitude)**2 for amplitude in self.state_vector]
        return random.choices(range(len(self.state_vector)), weights=probabilities, k=1)[0]