from polaris.harmonics import shcoeffs
import numpy as np

def test_multiply():
    x = shcoeffs.SHCoeffs(np.random.random((6,)))
    i = shcoeffs.SHCoeffs([np.sqrt(4*np.pi),0,0,0,0,0]) # Identity

    assert np.allclose((i*i).coeffs[:6], i.coeffs[:6]) 
    assert np.allclose((i*i).coeffs[7:], 0)

    assert np.allclose((x*i).coeffs[:6], x.coeffs[:6]) 
    assert np.allclose((x*i).coeffs[7:], 0)

def test_division():
    x = shcoeffs.SHCoeffs(np.random.random((6,)))
    y = shcoeffs.SHCoeffs(np.random.random((6,)))

    assert np.allclose((x*y/y).coeffs[:6], x.coeffs[:6]) 
    assert np.allclose((x*y/y).coeffs[7:], 0)

    assert np.allclose((x*y/x).coeffs[:6], y.coeffs[:6]) 
    assert np.allclose((x*y/x).coeffs[7:], 0)
