# Super Doodle - Quantum Architecture Simulator

A quantum circuit simulator built on [QuTiP](https://qutip.org/) for open quantum systems research.

## Project overview

Python library providing state-vector simulation, quantum gate operations, Hamiltonian evolution, and projective measurements.

### Files

- **quantum_simulator.py** — Main simulator. Uses QuTiP's `mesolve` for Hamiltonian evolution, projective measurement, and Bloch sphere visualization. Run this file directly for the built-in demo.
- **quantum_gates.py** — Quantum gate class hierarchy (Pauli X/Y/Z, Hadamard). Gate `apply()` methods are stubs for future implementation.
- **quantum_state.py** — Lightweight state vector wrapper with normalization and sampling.
- **quantum_circuit.py** — Simple circuit builder (gate list + run stub).

## How to run

```
python quantum_simulator.py
```

The "Run Simulator" workflow runs this automatically.

## Dependencies

- `qutip` — quantum systems simulation
- `numpy` — numerical arrays
- `matplotlib` — optional, used by Bloch sphere plotting

## Notes

- QuTiP 5 changed `mesolve` to keyword-only args for `c_ops`/`e_ops` — already patched.
- `quantum_gates.py` and `quantum_circuit.py` contain stub implementations ready to be fleshed out.

## User preferences

- Keep the existing Python project structure.
