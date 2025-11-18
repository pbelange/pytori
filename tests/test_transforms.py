# ============================================================
# pytest: pytori transforms vs ducktrack reference elements
# ============================================================
import numpy as np
import pytest

import ducktrack as dtk
import pytori.mathlib as mathlib
from pytori.series import FourierSeries
from pytori.tori import Torus
import pytori.transforms as transforms  # your drift, bend, multipole, etc.

# ============================================================
# Global reference particle ("on closed orbit")
# ============================================================

# Global energy template for all particles (7 TeV proton, nominal LHC scale)
particle_ref = dtk.TestParticles(
    p0c=7e12,  # 7 TeV
    x=0, px=0,
    y=0, py=0,
    zeta=0, pzeta=0,
)

# ============================================================
# Helpers
# ============================================================

def make_simple_torus(dim=3, amp=1e-3):
    """
    Build a minimal test Torus with one Fourier mode per plane:
        x(Θ) = A_x e^{iΘ_x}
        y(Θ) = A_y e^{iΘ_y}
        z(Θ) = A_z e^{iΘ_z}
    """
    coeff_x = {(1, 0, 0): amp * np.exp(1j * 0.1)}
    coeff_y = {(0, 1, 0): amp * np.exp(1j * 0.2)}
    coeff_z = {(0, 0, 1): amp * np.exp(1j * 0.3)}

    return Torus(
        x=FourierSeries(coeff_x, dim=dim, mp="numpy"),
        y=FourierSeries(coeff_y, dim=dim, mp="numpy"),
        z=FourierSeries(coeff_z, dim=dim, mp="numpy"),
        bet0=(1.0, 1.0, 1.0),
    )


def torus_to_particles(Psi, Tx, Ty, Tz):
    """
    Evaluate Torus over arrays of (Tx, Ty, Tz) and return ducktrack.TestParticles.
    Uses the global particle_ref as a template.
    """
    x = Psi.X(Tx, Ty, Tz)
    px = Psi.PX(Tx, Ty, Tz)
    y = Psi.Y(Tx, Ty, Tz)
    py = Psi.PY(Tx, Ty, Tz)
    zeta = Psi.Z(Tx, Ty, Tz)
    pzeta = Psi.PZ(Tx, Ty, Tz)

    return dtk.TestParticles(
        p0c = particle_ref.p0c,
        x=x.ravel(), px=px.ravel(),
        y=y.ravel(), py=py.ravel(),
        zeta=zeta.ravel(), pzeta=pzeta.ravel(),
    )


def compare_torus_vs_ducktrack(Psi_new, dtk_final, Tx, Ty, Tz, rtol=1e-12, atol=1e-14):
    """
    Compare pytori transform results to ducktrack reference at given arrays of phases.
    """
    x_t = Psi_new.X(Tx, Ty, Tz).ravel()
    px_t = Psi_new.PX(Tx, Ty, Tz).ravel()
    y_t = Psi_new.Y(Tx, Ty, Tz).ravel()
    py_t = Psi_new.PY(Tx, Ty, Tz).ravel()
    z_t = Psi_new.Z(Tx, Ty, Tz).ravel()
    pz_t = Psi_new.PZ(Tx, Ty, Tz).ravel()

    np.testing.assert_allclose(x_t, dtk_final.x, rtol=rtol, atol=atol, err_msg="Mismatch in x")
    np.testing.assert_allclose(px_t, dtk_final.px, rtol=rtol, atol=atol, err_msg="Mismatch in px")
    np.testing.assert_allclose(y_t, dtk_final.y, rtol=rtol, atol=atol, err_msg="Mismatch in y")
    np.testing.assert_allclose(py_t, dtk_final.py, rtol=rtol, atol=atol, err_msg="Mismatch in py")
    np.testing.assert_allclose(z_t, dtk_final.zeta, rtol=rtol, atol=atol, err_msg="Mismatch in zeta")
    np.testing.assert_allclose(pz_t, dtk_final.pzeta, rtol=rtol, atol=atol, err_msg="Mismatch in pzeta")


# ============================================================
# Common 3D phase grid
# ============================================================

@pytest.fixture(scope="module")
def phase_grid():
    """3D phase mesh for evaluating the torus."""
    Tx = np.linspace(0, 2 * np.pi, 5)
    Ty = np.linspace(0, 2 * np.pi, 5)
    Tz = np.linspace(0, 2 * np.pi, 5)
    return np.meshgrid(Tx, Ty, Tz, indexing="ij")


# ============================================================
# Tests
# ============================================================

@pytest.mark.parametrize("ds", [0.05, 0.1])
def test_drift_vs_ducktrack(phase_grid, ds):
    """Compare pytori.drift with ducktrack.DriftExact."""
    Tx, Ty, Tz = phase_grid
    Psi = make_simple_torus()

    # --- Reference: ducktrack
    dtk_particles = torus_to_particles(Psi, Tx, Ty, Tz)
    el_dtk = dtk.elements.Drift(length=ds)  # exact symplectic drift
    el_dtk.track(dtk_particles)

    # --- pytori: drift map
    Psi_new = transforms.drift(Psi, ds=ds, particle_ref=particle_ref)
    compare_torus_vs_ducktrack(Psi_new, dtk_particles, Tx, Ty, Tz)


@pytest.mark.parametrize(
    "knl, ksl",
    [
        ([0, 1.0], [0, 0.0]),           # pure normal quadrupole
        ([0, 0.5, 0.1], [0, 0.2, -0.8]), # higher-order normal multipole
        ([0, 1.0], [0, 1.0]),           # combined normal+skew
        ([0, 0.0], [0, 1.0]),           # pure skew quadrupole
    ],
)
def test_multipole_vs_ducktrack(phase_grid, knl, ksl):
    """Compare pytori.multipole with ducktrack.Multipole for both knl and ksl."""
    Tx, Ty, Tz = phase_grid
    Psi = make_simple_torus()

    # Create DuckTrack particles
    dtk_particles = torus_to_particles(Psi, Tx, Ty, Tz)

    # DuckTrack element
    el_dtk = dtk.elements.Multipole(knl=knl, ksl=ksl)
    el_dtk.track(dtk_particles)

    # pytori equivalent
    Psi_new = transforms.multipole(Psi, knl=knl, ksl=ksl)

    # Compare
    compare_torus_vs_ducktrack(Psi_new, dtk_particles, Tx, Ty, Tz)


@pytest.mark.parametrize("k0, h", [(0.01, 0.01), (-0.02, -0.01)])
def test_bend_vs_ducktrack(phase_grid, k0, h):
    """Compare pytori.bend with ducktrack.Bend."""
    Tx, Ty, Tz = phase_grid
    Psi = make_simple_torus()

    dtk_particles = torus_to_particles(Psi, Tx, Ty, Tz)

    # There is no thin-bend per say. So we model it as a multipole with ref trajectory curvature and frozen y-motion. 
    #------------------------------------
    # Save original y, py to restore after tracking
    y0 = dtk_particles.y.copy()
    py0 = dtk_particles.py.copy()

    el_dtk = dtk.elements.Multipole(knl=[k0,k0*h],hxl=h)
    el_dtk.track(dtk_particles)

    dtk_particles.y = y0
    dtk_particles.py = py0
    #------------------------------------
    
    Psi_new = transforms.bend(Psi, k0=k0,h=h, particle_ref=particle_ref)
    compare_torus_vs_ducktrack(Psi_new, dtk_particles, Tx, Ty, Tz)