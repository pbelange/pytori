
from pytori.tori import Torus
import pytori.mathlib as mathlib
import pytori.normalform as normalform


def drift(Psi: Torus, ds=0, particle_ref=None, beta_rel=None, order=20):
    """
    Apply a symplectic drift transformation to a Torus:
        H = pz - δ + (px² + py²) / [2(1 + δ)]
    using truncated Gegenbauer expansion for chromatic coupling.

    Parameters
    ----------
    Psi : Torus
        The Torus to transform (NormalForm or Fourier representation).
    ds : float
        Drift length.
    particle_ref : object, optional
        Reference particle, used to extract β_rel if not given.
    beta_rel : float, optional
        Relativistic β. Required if not inferrable from `particle_ref`.
    order : int, optional
        Truncation order for Gegenbauer expansion.
    """

    mp = mathlib.import_mathlib(Psi.mp)

    # Extracting projections
    Psix, Psiy, Psiz = Psi.x, Psi.y, Psi.z
    betx0, bety0, betz0 = Psi.betx0, Psi.bety0, Psi.betz0

    # Relativistic β
    if Psiz is not None and beta_rel is None and particle_ref is not None:
        try:
            beta_rel = getattr(particle_ref, "beta0")[0]
        except:
            beta_rel = getattr(particle_ref, "beta0")
    assert beta_rel is not None or Psiz is None, "beta_rel must be provided for z-plane"

    
    # Canonical momenta
    Px = mathlib.Pj(Psix, betx0, mp=mp) if Psix else 0
    Py = mathlib.Pj(Psiy, bety0, mp=mp) if Psiy else 0
    Pz = mathlib.Pj(Psiz, betz0, mp=mp) if Psiz else 0

    #------------------------------------------------------------
    # Chromatic factors (symplectic truncation)
    #------------------------------------------------------------
    if Psiz:
        Pz_pwrs = mathlib.list_powers(beta_rel * Pz, order)
        Cz      = mathlib._truncated_gegenbauer_series(Pz_pwrs, +0.5, beta_rel, mp)
        dCz     = mathlib._d_truncated_gegenbauer_series(Pz_pwrs, +0.5, beta_rel, mp)
        dCzinv  = mathlib._d_truncated_gegenbauer_series(Pz_pwrs, -0.5, beta_rel, mp)
    else:
        Cz, dCz, dCzinv = 1, 0, 0

    #------------------------------------------------------------
    # Lie map (exact to truncation order)
    #------------------------------------------------------------
    _x = Psix + ds / mp.sqrt(betx0) * (Px * Cz) if Psix else None
    _y = Psiy + ds / mp.sqrt(bety0) * (Py * Cz) if Psiy else None

    if Psiz:
        zkick = 1 - dCzinv + dCz * (Px * Px + Py * Py) / 2
        _z = Psiz + ds / mp.sqrt(betz0) * zkick
    else:
        _z = None

    #------------------------------------------------------------
    # Return transformed Torus
    #------------------------------------------------------------
    return Torus(x=_x, y=_y, z=_z,bet0=(betx0, bety0, betz0))




def bend(Psi: Torus, k0=None, h=None, particle_ref=None, beta_rel=None, order=20):
    """
    Apply a thin-dipole transformation to a Torus:
        H = -h * x * (1 + δ) + k0 * (x + h x² / 2)

    Uses truncated Gegenbauer expansion for chromatic coupling (α = −½).
    """

    mp = mathlib.import_mathlib(Psi.mp)

    Psix, Psiy, Psiz = Psi.x, Psi.y, Psi.z
    betx0, bety0, betz0 = Psi.betx0, Psi.bety0, Psi.betz0

    # Relativistic β
    if Psiz is not None and beta_rel is None and particle_ref is not None:
        try:
            beta_rel = getattr(particle_ref, "beta0")[0]
        except:
            beta_rel = getattr(particle_ref, "beta0")
    assert beta_rel is not None or Psiz is None, "beta_rel must be provided for z-plane"

    # Field geometry
    if h is None and k0 is not None:
        h = k0
    elif h is not None and k0 is None:
        k0 = h
    elif h is None and k0 is None:
        raise ValueError("Either k0 or h must be provided")

    # Coordinates
    X  = mathlib.Qj(Psix, betx0, mp=mp) if Psix else 0
    Pz = mathlib.Pj(Psiz, betz0, mp=mp) if Psiz else 0

    # Chromatic factors
    if Psiz:
        Pz_pwrs = mathlib.list_powers(beta_rel * Pz, order)
        Czinv      = mathlib._truncated_gegenbauer_series(Pz_pwrs, -0.5, beta_rel, mp)
        dCzinv_dPz = mathlib._d_truncated_gegenbauer_series(Pz_pwrs, -0.5, beta_rel, mp)
    else:
        Czinv, dCzinv_dPz = 1, 0

    # Thin-lens map
    _x = Psix + 1j * mp.sqrt(betx0) * (k0 * (1 + h * X) - h * Czinv) if Psix else None
    _y = Psiy if Psiy else None
    _z = Psiz - h / mp.sqrt(betz0) * dCzinv_dPz * X if Psiz else None

    return Torus(x=_x, y=_y, z=_z,bet0=(betx0, bety0, betz0))


def multipole(Psi: Torus, knl=None, ksl=None):
    """
    Apply a thin-multipole transformation to a Torus:
        H = Re[(knl + i * ksl) * (x + i y)^(n+1) / (n+1)!]

    Uses Horner-like recursion (see xsuite) for numerical stability.
    """
    mp = mathlib.import_mathlib(Psi.mp)
    Psix, Psiy, Psiz = Psi.x, Psi.y, Psi.z
    betx0, bety0, betz0 = Psi.betx0, Psi.bety0, Psi.betz0

    knl = knl if knl is not None else []
    ksl = ksl if ksl is not None else []
    order = max(len(knl), len(ksl)) - 1
    knl = mathlib._arrayofsize(knl, order + 1)
    ksl = mathlib._arrayofsize(ksl, order + 1)

    # Extract position series
    X = mathlib.Qj(Psix, betx0, mp=mp) if Psix else 0
    Y = mathlib.Qj(Psiy, bety0, mp=mp) if Psiy else 0


    # Following xsuite's implementation, we use a Horner-like recursion
    dpx = knl[order]
    dpy = ksl[order]
    for ii in range(order, 0, -1):
        zre = (dpx * X - dpy * Y) / ii
        zim = (dpx * Y + dpy * X) / ii
        dpx = knl[ii - 1] + zre
        dpy = ksl[ii - 1] + zim
    dpx = -1 * dpx
    dpy =  1 * dpy

    _x = Psix - 1j * mp.sqrt(betx0) * dpx if Psix else None
    _y = Psiy - 1j * mp.sqrt(bety0) * dpy if Psiy else None
    _z = Psiz if Psiz else None

    return Torus(x=_x, y=_y, z=_z,bet0=(betx0, bety0, betz0))



def linear_map(Psi: Torus,
               Qvec=None, lambda_plus=None, lambda_minus=None, W_matrix=None,
               Lp_list=None, Lm_list=None, W_list=None,
               U_matrix=None, V_matrix=None):
    """
    Apply a linear map to (ψ_x, ψ_y, ψ_ζ) in the complex basis Ψ, via
        Ψ' = U Ψ + V Ψ*
    where, if U,V are not provided:
        U =   Lp E Lp^† - Lm E^* Lm^†
        V = - Lp E Lm^T + Lm E^* Lp^T
    with E = diag(exp(i*2π Q_j)).
    """

    mp = mathlib.import_mathlib(Psi.mp)

    # Active inputs (keep x,y,z ordering)
    Psix, Psiy, Psiz = Psi.x, Psi.y, Psi.z
    psi_list    = [Psix, Psiy, Psiz]
    active_dims = [i for i, psi in enumerate(psi_list) if psi is not None]
    psi_vec     = [psi_list[i] for i in active_dims]
    dim = len(psi_vec)

    # Decide whether we build U,V or use provided
    need_UV = (U_matrix is None and V_matrix is None)

    if need_UV:
        assert Qvec is not None, "Qvec (phase advances) must be provided when U,V are not."
        assert len(Qvec) >= dim, f"Expected at least {dim} phase advances"

        U, V = normalform.construct_UV(
            Qvec=Qvec,
            lambda_plus=lambda_plus,
            lambda_minus=lambda_minus,
            W_matrix=W_matrix,
            Lp_list=Lp_list,
            Lm_list=Lm_list,
            W_list=W_list,
            mp=mp
        )
    else:
        assert (U_matrix is not None) and (V_matrix is not None), \
            "Both U_matrix and V_matrix must be provided"
        try:
            U = U_matrix.tolist()
            V = V_matrix.tolist()
        except Exception:
            U = U_matrix
            V = V_matrix
        assert len(U) == dim and len(V) == dim, "U,V must match active dimension"
        assert all(len(row) == dim for row in U) and all(len(row) == dim for row in V), \
            "U,V must be square (dim x dim)"

    # Apply Ψ' = U Ψ + V Ψ*
    psi_out = [0 for _ in range(dim)]
    for i in range(dim):
        total = 0
        for k in range(dim):
            total += U[i][k] * psi_vec[k]
            total += V[i][k] * psi_vec[k].conjugate()
        psi_out[i] = total

    # Repack into (x,y,z) slots
    result = [None, None, None]
    for loc, idx in enumerate(active_dims):
        result[idx] = psi_out[loc]

    return Torus(x=result[0], y=result[1], z=result[2],
                 bet0=(Psi.betx0, Psi.bety0, Psi.betz0))



def phys2norm(Psi: Torus,
              lambda_plus=None, lambda_minus=None, W_matrix=None,
              nemitt_x=None, nemitt_y=None, nemitt_z=None,
              particle_on_co=None, beta_rel=None, gamma_rel=None):
    """
    Apply normalization transformation to coupled phase space variables (ψ_x, ψ_y, ψ_ζ),
    converting them into decoupled (normalized) variables (ψ̃_x, ψ̃_y, ψ̃_ζ).
    """

    mp = mathlib.import_mathlib(Psi.mp)

    # Decide λ⁺, λ⁻
    if W_matrix is not None:
        assert lambda_plus is None and lambda_minus is None, "Provide either W_matrix or λ±, not both."
        lambda_plus, lambda_minus = mathlib.W_to_lambda(W_matrix)
    else:
        assert lambda_plus is not None and lambda_minus is not None, "lambda_plus and lambda_minus must be provided."

    # Extract planes
    Psix, Psiy, Psiz = Psi.x, Psi.y, Psi.z
    psi_list = [Psix, Psiy, Psiz]
    active_dims = [i for i, psi in enumerate(psi_list) if psi is not None]
    dim = len(active_dims)

    assert lambda_plus.shape == (dim, dim), \
        f"Expected lambda matrices of shape ({dim},{dim}), got {lambda_plus.shape}"

    # Closed orbit + geometric emittances
    co, geo = mathlib.co_geo_normalization(
        nemitt_x=nemitt_x, nemitt_y=nemitt_y, nemitt_z=nemitt_z,
        particle_on_co=particle_on_co, beta_rel=beta_rel, gamma_rel=gamma_rel
    )

    # Subtract closed orbit
    psi_vec = [psi_list[idx] - co[idx] for idx in active_dims]

    # Normalization transformation
    psi_tilde = [0] * dim
    for i in range(dim):
        for j in range(dim):
            psi_tilde[i] += lambda_plus[j, i].conjugate() * psi_vec[j] \
                            - lambda_minus[j, i] * psi_vec[j].conjugate()

    # Emittance rescaling and repack
    result = [None, None, None]
    for k, idx in enumerate(active_dims):
        result[idx] = psi_tilde[k] / mp.sqrt(geo[idx])

    return Torus(x=result[0], y=result[1], z=result[2],
                 bet0=(Psi.betx0, Psi.bety0, Psi.betz0))




def norm2phys(Psi: Torus,
              lambda_plus=None, lambda_minus=None, W_matrix=None,
              nemitt_x=None, nemitt_y=None, nemitt_z=None,
              particle_on_co=None, beta_rel=None, gamma_rel=None):
    """
    Apply inverse normalization transformation to decoupled phase space variables (ψ̃_x, ψ̃_y, ψ̃_ζ),
    reconstructing the coupled (physical) variables (ψ_x, ψ_y, ψ_ζ).
    """

    mp = mathlib.import_mathlib(Psi.mp)

    # Decide λ⁺, λ⁻
    if W_matrix is not None:
        assert lambda_plus is None and lambda_minus is None, "Provide either W_matrix or λ±, not both."
        lambda_plus, lambda_minus = mathlib.W_to_lambda(W_matrix)
    else:
        assert lambda_plus is not None and lambda_minus is not None, "lambda_plus and lambda_minus must be provided."

    # Extract planes
    Psix, Psiy, Psiz = Psi.x, Psi.y, Psi.z
    psi_list = [Psix, Psiy, Psiz]
    active_dims = [i for i, psi in enumerate(psi_list) if psi is not None]
    dim = len(active_dims)

    assert lambda_plus.shape == (dim, dim), \
        f"Expected lambda matrices of shape ({dim},{dim}), got {lambda_plus.shape}"

    # Closed orbit + geometric emittances
    co, geo = mathlib.co_geo_normalization(
        nemitt_x=nemitt_x, nemitt_y=nemitt_y, nemitt_z=nemitt_z,
        particle_on_co=particle_on_co, beta_rel=beta_rel, gamma_rel=gamma_rel
    )

    # Emittance rescaling
    psi_vec = [psi_list[idx] * mp.sqrt(geo[idx]) for idx in active_dims]

    # Denormalization transformation
    psi_phys = [0] * dim
    for i in range(dim):
        for j in range(dim):
            psi_phys[i] += lambda_plus[i, j] * psi_vec[j] \
                           + lambda_minus[i, j] * psi_vec[j].conjugate()

    # Add closed orbit and repack
    result = [None, None, None]
    for k, idx in enumerate(active_dims):
        result[idx] = psi_phys[k] + co[idx]

    return Torus(x=result[0], y=result[1], z=result[2],
                 bet0=(Psi.betx0, Psi.bety0, Psi.betz0))