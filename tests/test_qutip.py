import qutip as qt
import numpy as np
import pytest

def test_qutip_import():
    """Verify QuTiP is installed."""
    assert qt.__version__ is not None
    print(f'✅ QuTiP {qt.__version__} imported successfully')

def test_harmonic_oscillator():
    """Test basic mesolve on harmonic oscillator."""
    N = 15
    a = qt.destroy(N)
    H = a.dag() * a
    psi0 = qt.coherent(N, 2.0)
    tlist = np.linspace(0, 5, 50)
    result = qt.mesolve(H, psi0, tlist, c_ops=[], e_ops=[a.dag() * a])
    expect = result.expect[0]
    assert len(expect) == len(tlist)
    assert abs(expect[0] - 4.0) < 0.5, f'Initial expectation should be near 4, got {expect[0]}'
    print('✅ QuTiP dynamics test passed')}