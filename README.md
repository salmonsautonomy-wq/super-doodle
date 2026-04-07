# Super Doodle - Quantum Architecture Simulator

A quantum circuit simulator implementing state-vector simulation for quantum computing algorithms.

## Overview

This project focuses on **Quantum Architecture** research and provides tools for:
- Quantum gate implementations (Pauli, Hadamard, rotations, controlled gates)
- Quantum state management and measurements
- Quantum circuit construction and execution
- State vector simulation

## Files

- **quantum_gates.py** - Quantum gate definitions and matrix representations
- **quantum_state.py** - Quantum state management with superposition and entanglement
- **quantum_circuit.py** - Circuit builder interface
- **quantum_simulator.py** - State vector simulator for circuit execution

## Quick Start

```python
from quantum_circuit import QuantumCircuit
from quantum_simulator import QuantumSimulator

# Create a 2-qubit circuit
circuit = QuantumCircuit(2)
circuit.h(0)  # Hadamard on qubit 0
circuit.cnot(0, 1)  # CNOT with control=0, target=1

# Run simulator
simulator = QuantumSimulator()
results = simulator.run(circuit, shots=1000)
simulator.print_results(results, shots=1000)
