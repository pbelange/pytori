# ============================================================
# tests/test_torus.py
# ============================================================

import pytest
import numpy as np

from pytori.tori import Torus
from pytori.series import FourierSeries, NormalFormSeries


# ============================================================
# 1. Basic construction & dimension inference
# ============================================================

def test_torus_init_single_plane_1d():
    """Torus with a single 1D FourierSeries should infer dim=1."""
    x_series = FourierSeries({(1,): 1.0}, dim=1, mp="numpy")
    T = Torus(x=x_series, mp="numpy")

    assert T.dim == 1
    assert T.series_class is FourierSeries
    assert T.mp.name == "numpy"
    assert T.x is x_series
    assert T.y is None
    assert T.z is None
    assert T.needs_refresh is True  # cache not built yet


# ============================================================
# 2. Cache behavior & needs_refresh flag
# ============================================================

def test_update_cache_and_needs_refresh():
    """_update_cache should populate Ax/nx and clear the refresh flag."""
    x_series = FourierSeries({(1,): 2.0}, dim=1, mp="numpy")
    T = Torus(x=x_series, mp="numpy")

    # Initially needs_refresh is True
    assert T.needs_refresh is True

    # Build cache
    T._update_cache()
    assert T.needs_refresh is False

    # Ax/nx should reflect the x_series coefficients
    assert T._Ax == list(x_series.coeffs.values())
    assert T._nx == list(x_series.coeffs.keys())

    # Changing x should toggle refresh
    T.x = FourierSeries({(1,): 3.0}, dim=1, mp="numpy")
    assert T.needs_refresh is True


# ============================================================
# 3. Evaluation (1D) and coordinate accessors
# ============================================================

def test_eval_X_PX_1d():
    """
    For x(θ) = e^{i θ}, we expect:
        X(θ)  = cos(θ)
        PX(θ) = -sin(θ)
    """
    x_series = FourierSeries({(1,): 1.0}, dim=1, mp="numpy")
    T = Torus(x=x_series, mp="numpy")

    # θ = 0
    X0 = T.X(Tx=0.0)
    PX0 = T.PX(Tx=0.0)
    assert np.isclose(X0, 1.0)
    assert np.isclose(PX0, 0.0)

    # θ = π/2
    theta = np.pi / 2
    X = T.X(Tx=theta)
    PX = T.PX(Tx=theta)
    assert np.isclose(X, 0.0, atol=1e-12)
    assert np.isclose(PX, -1.0, atol=1e-12)


def test_eval_raises_for_missing_plane():
    """Calling eval on a missing plane should raise a ValueError."""
    x_series = FourierSeries({(1,): 1.0}, dim=1, mp="numpy")
    T = Torus(x=x_series, mp="numpy")

    with pytest.raises(ValueError):
        T.eval("y", Tx=0.0, Ty=0.0, Tz=0.0)


# ============================================================
# 4. Evaluation (2D) – X depends on Tx, Y on Ty
# ============================================================

def test_eval_X_Y_2d():
    """
    Construct a 2D torus:
      x(Θx,Θy) = e^{i Θx}
      y(Θx,Θy) = e^{i Θy}
    Then:
      X(Tx,Ty) ~ cos(Tx)
      Y(Tx,Ty) ~ cos(Ty)
    """
    x_series = FourierSeries({(1, 0): 1.0}, dim=2, mp="numpy")
    y_series = FourierSeries({(0, 1): 1.0}, dim=2, mp="numpy")
    T = Torus(x=x_series, y=y_series, mp="numpy")

    assert T.dim == 2

    Tx = 0.3
    Ty = 1.1

    X = T.X(Tx=Tx, Ty=Ty)
    Y = T.Y(Tx=Tx, Ty=Ty)

    assert np.isclose(X, np.cos(Tx), atol=1e-12)
    assert np.isclose(Y, np.cos(Ty), atol=1e-12)


# ============================================================
# 5. Dephasing
# ============================================================

def test_dephase_changes_coefficients():
    """
    Dephasing a 1D torus should multiply coefficients by e^{i k * phi}.
    For k=1, phi=π/2, the new coeff is 1 * e^{iπ/2} = i.
    """
    x_series = FourierSeries({(1,): 1.0}, dim=1, mp="numpy")
    T = Torus(x=x_series, mp="numpy")

    phi = np.pi / 2
    T_phi = T.dephase(phi)

    # Original coeff
    a0 = T.x.coeffs[(1,)]
    # Dephased coeff
    a1 = T_phi.x.coeffs[(1,)]

    # |a1| == |a0|
    assert np.isclose(np.abs(a1), np.abs(a0))
    # phase difference ~ phi
    phase_diff = np.angle(a1) - np.angle(a0)
    # modulo 2π comparison
    assert np.isclose(np.mod(phase_diff, 2 * np.pi), np.mod(phi, 2 * np.pi), atol=1e-8)


# ============================================================
# 6. Collapse – precondition & simple NormalForm case
# ============================================================

def test_collapse_requires_normalform():
    """Collapse should fail for a Torus built from FourierSeries."""
    x_series = FourierSeries({(1,): 1.0}, dim=1, mp="numpy")
    T = Torus(x=x_series, mp="numpy")
    with pytest.raises(AssertionError):
        T.collapse(I=0.5)


def test_collapse_normalform_1d():
    """
    Single-plane NormalFormSeries:
      coeff for (m,n) = (1,0) is 1.0

    Collapse at I=0.5 should give Fourier coefficient at k=1 with
    amplitude (2I)^{(m+n)/2} = (1)^{1/2} = 1.
    """
    # NormalFormSeries dim must be even; here dim=2 for one (ρ,ρ*) pair
    nf = NormalFormSeries({(1, 0): 1.0}, dim=2, max_order=2, mp="numpy")
    T_nf = Torus(x=nf, mp="numpy", series_class=NormalFormSeries)

    T_coll = T_nf.collapse(I=0.5, max_order=1, max_terms=10, numerical_tol=None, mp="numpy")
    assert isinstance(T_coll.x, FourierSeries)
    assert T_coll.x.dim == 1

    # Expect A_(1) ≈ 1.0
    assert np.isclose(T_coll.x.coeffs[(1,)], 1.0, atol=1e-12)


# ============================================================
# 7. Actions & quadratic invariants
# ============================================================

def test_Jx_matches_half_norm_squared():
    """
    For x(θ) = A e^{i θ}, Jx should be 0.5 * |A|^2.
    """
    A = 2.0 + 1.0j
    x_series = FourierSeries({(1,): A}, dim=1, mp="numpy")
    T = Torus(x=x_series, mp="numpy")

    Jx = T.Jx  # this triggers _update_cache internally
    expected = 0.5 * np.abs(A) ** 2
    assert np.isclose(Jx, expected, atol=1e-12)


def test_Ix_R_R_T_are_scalars():
    """
    We don't assert the exact value of Ix (depends on poincare_avg),
    but we do check it returns a scalar and that R, R_T are scalar too.
    """
    x_series = FourierSeries({(1,): 1.0}, dim=1, mp="numpy")
    T = Torus(x=x_series, mp="numpy")

    # Ix may require poincare_avg details; we only check shape/type
    Ix = T.Ix
    assert np.ndim(Ix) == 0

    # R and R_T should also be scalar (real or complex)
    R = T.R
    R_T = T.R_T
    assert np.ndim(R) == 0
    assert np.ndim(R_T) == 0


# ============================================================
# 8. Copy
# ============================================================

def test_torus_copy_preserves_structure():
    """copy() should produce another Torus with equivalent series content."""
    x_series = FourierSeries({(1,): 1.0}, dim=1, mp="numpy")
    T = Torus(x=x_series, bet0=(1.0, 2.0, 3.0), mp="numpy")

    T2 = T.copy()
    assert isinstance(T2, Torus)
    assert T2 is not T
    assert T2.x is not T.x  # new series object
    assert T2.x.coeffs == T.x.coeffs
    assert (T2.betx0, T2.bety0, T2.betz0) == (T.betx0, T.bety0, T.betz0)


# ============================================================
# 9. Representation
# ============================================================

def test_torus_repr_safe():
    """__repr__ should return a string and not raise, even if display backend changes."""
    x_series = FourierSeries({(1,): 1.0}, dim=1, mp="numpy")
    T = Torus(x=x_series, mp="numpy")
    s = repr(T)
    assert isinstance(s, str)
    # allow empty or pretty output; we only require that it's a string
