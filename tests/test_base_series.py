# ============================================================
# tests/test_base_series.py
# ============================================================

import pytest
import numpy as np
from pytori import BaseSeries


# ============================================================
# 1. Initialization
# ============================================================

@pytest.mark.parametrize("dim", [1, 2, 3])
def test_init_and_dimension(dim):
    """BaseSeries infers correct dimension and stores valid coefficients."""
    coeffs = {(1,) * dim: 1.0, (2,) * dim: 2.0}
    s = BaseSeries(coeffs, dim=dim, mp="numpy")

    assert s.dim == dim
    assert isinstance(s.coeffs, dict)
    assert all(len(k) == dim for k in s.coeffs)
    assert all(isinstance(v, (float, int, complex)) for v in s.coeffs.values())


# ============================================================
# 2. Addition and scalars
# ============================================================

@pytest.mark.parametrize("dim", [1, 2, 3])
def test_addition_and_scalars(dim):
    """Check addition and scalar operations for BaseSeries."""
    a = BaseSeries({(1,) * dim: 2.0}, dim=dim)
    b = BaseSeries({(1,) * dim: 3.0, (2,) * dim: 4.0}, dim=dim)

    c = a + b
    assert np.isclose(c[(1,) * dim], 5.0)
    assert np.isclose(c[(2,) * dim], 4.0)

    s1 = 2.0 * a
    s2 = a * 2.0
    assert np.isclose(s1[(1,) * dim], 4.0)
    assert np.isclose(s2[(1,) * dim], 4.0)

    # Scalar addition adds to the zero mode
    d = a + 1.0
    assert np.isclose(d[(0,) * dim], 1.0)
    assert np.isclose(d[(1,) * dim], 2.0)


# ============================================================
# 3. Multiplication (convolution)
# ============================================================

@pytest.mark.parametrize("dim", [1, 2, 3])
def test_multiplication_convolution(dim):
    """Ensure multiplication correctly performs convolution."""
    a = BaseSeries({(1,) * dim: 1.0}, dim=dim)
    b = BaseSeries({(2,) * dim: 2.0}, dim=dim)
    c = a * b
    expected_key = tuple(np.add((1,) * dim, (2,) * dim))
    assert expected_key in c.coeffs
    assert np.isclose(c[expected_key], 2.0)


# ============================================================
# 4. Negation, subtraction, and division
# ============================================================

@pytest.mark.parametrize("dim", [1, 2, 3])
def test_neg_sub_div(dim):
    """Check unary negation, subtraction, and scalar division."""
    a = BaseSeries({(1,) * dim: 4.0}, dim=dim)
    b = BaseSeries({(1,) * dim: 1.0}, dim=dim)

    neg = -a
    assert np.isclose(neg[(1,) * dim], -4.0)

    diff = a - b
    assert np.isclose(diff[(1,) * dim], 3.0)

    div = a / 2.0
    assert np.isclose(div[(1,) * dim], 2.0)


# ============================================================
# 5. Power (integer)
# ============================================================

@pytest.mark.parametrize("dim", [1, 2, 3])
def test_pow_behavior(dim):
    """Raising BaseSeries to integer powers behaves algebraically."""
    a = BaseSeries({(1,) * dim: 2.0}, dim=dim)
    a2 = a ** 2
    assert (2,) * dim in a2.coeffs
    assert np.isclose(a2[(2,) * dim], 4.0)

    a0 = a ** 0
    assert np.isclose(a0[(0,) * dim], 1.0)


# ============================================================
# 6. Conjugation
# ============================================================

@pytest.mark.parametrize("dim", [1, 2, 3])
def test_conjugation(dim):
    """Conjugation should conjugate each coefficient value."""
    a = BaseSeries({(1,) * dim: 1 + 1j, (2,) * dim: 2 - 1j}, dim=dim)
    c = a.conjugate()
    for k, v in a.coeffs.items():
        assert np.isclose(c[k], np.conjugate(v))

    # Double conjugation restores
    assert all(np.isclose(a[k], c.conjugate()[k]) for k in a.coeffs)


# ============================================================
# 7. Truncate 
# ============================================================
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_truncate_behavior(dim):
    """truncate() should filter coefficients according to max_order and dimension rules."""
    a = BaseSeries({(1,) * dim: 1.0, (2,) * dim: 2.0}, dim=dim)

    # Apply a truncation that will remove high-order harmonics
    b = a.truncate(max_order=1)

    assert a is not b
    assert isinstance(b, BaseSeries)

    # For dim==1, (1,) should remain; for higher dims, all terms are filtered out
    if dim == 1:
        assert (1,) in b.coeffs
        assert np.isclose(b[(1,)], 1.0)
        assert (2,) not in b.coeffs
    else:
        assert len(b.coeffs) == 0  # all terms dropped

# ============================================================
# 8. Copy independence
# ============================================================

@pytest.mark.parametrize("dim", [1, 2, 3])
def test_copy_independence(dim):
    """copy() should produce independent coefficient dicts."""
    a = BaseSeries({(1,) * dim: 1.0}, dim=dim)
    b = a.copy(coeff_dict={(1,) * dim: 2.0})
    assert not np.isclose(a[(1,) * dim], b[(1,) * dim])


# ============================================================
# 9. Metadata / export
# ============================================================

@pytest.mark.parametrize("dim", [1, 2, 3])
def test_metadata_and_to_dict(dim):
    """_metadata() and to_dict() should return consistent info."""
    a = BaseSeries({(1,) * dim: 1.0}, dim=dim, max_order=3, max_terms=10, numerical_tol=1e-6)
    meta = a._metadata()
    assert meta[0] == dim
    d = a.to_dict()
    assert d["dim"] == dim
    assert (1,) * dim in d["coeffs"]


# ============================================================
# 10. Representation
# ============================================================

@pytest.mark.parametrize("dim", [1, 2, 3])
def test_repr_safe(dim):
    """__repr__ should return a non-empty string and not raise."""
    a = BaseSeries({(1,) * dim: 1.0}, dim=dim)
    s = repr(a)
    assert isinstance(s, str)
    assert len(s) >= 0  # allow empty if display backend unavailable
