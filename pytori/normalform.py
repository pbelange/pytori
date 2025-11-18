import numpy as np
import math
import itertools

import pytori as pt
import pytori.mathlib as mathlib



def xreplace(global_subs,num_subs):
    numeric  = dict({**global_subs,**num_subs})
    for k in numeric:
        if hasattr(numeric[k],'xreplace'):
            numeric[k] = numeric[k].xreplace(numeric).evalf()
    return numeric

def symbolic_normalform(order, dim, coeff_symbol='a'):
    """
    Create a symbolic multivariate BaseSeries (or NormalFormSeries-like) object
    with coefficients a_{n1 n2 ...}, for all |n| <= order.

    Parameters
    ----------
    coeff_symbol : str, 
        Base symbol for coefficient names (e.g. 'a' → a10, a21, etc.).
    dim : int, 
        Number of base variables (e.g. [ρ, ρ*]).
    order : int, 
        Maximum total order (sum of indices <= order).

    Returns
    -------
    BaseSeries
        Instance populated with symbolic coefficients.
    """
    import sympy as sp
    # Create symbolic coefficient dictionary
    coeff_dict = {}
    for k in itertools.product(range(order + 1), repeat=dim):
        if sum(k) <= order:
            name = coeff_symbol + ''.join(map(str, k))
            coeff_dict[k] = sp.Symbol(name)

    return pt.NormalFormSeries(coeff_dict=coeff_dict, dim=dim, max_order=order,mp='sympy')


def symbolic_detuning(order, dim, coeff_symbol='ω_x'):
    """
    Create a symbolic NormalFormSeries for detuning with coefficients ω_x<j1...jp>,
    placed on the full 2n multi-index with per-plane equality constraints:
        k = (j1, j1, j2, j2, ..., jp, jp),  sum(jm) <= order.

    Parameters
    ----------
    order : int
        Maximum total order in actions (sum of per-plane powers <= order).
    dim : int
        Full phase-space dimension: 2 -> 1 plane, 4 -> 2 planes, 6 -> 3 planes, ...
    coeff_symbol : str, default='ω_x'
        Base symbol for coefficient names (e.g. 'ω_x' -> ω_x0, ω_x1, ω_x10, ...).

    Returns
    -------
    pt.NormalFormSeries
        Symbolic series with sympy backend, indices length == dim, and
        per-plane pair-equality enforced.
    """
    import sympy as sp
    assert dim % 2 == 0 and dim > 0, "dim must be 2 * (# of planes)"
    n_planes = dim // 2

    coeff_dict = {}
    # action-powers vector a = (j1,..., j_{n_planes})
    for a in itertools.product(range(order + 1), repeat=n_planes):
        if sum(a) <= order:
            # expand to full 2n index with pair equality (p,p) per plane
            k = tuple(v for p in a for v in (p, p))  # (j1,j1,j2,j2,...)
            # name e.g. ω_x + "10" for a=(1,0); ω_x + "1" for a=(1,) etc.
            name = coeff_symbol + ''.join(map(str, a))
            coeff_dict[k] = sp.Symbol(name, real=True)

    return pt.NormalFormSeries(coeff_dict=coeff_dict,dim=dim,max_order=order, mp='sympy')



def rho_transform(plane_j, omega_j, order, mp='numpy'):
    """
    Build f_j(ρ,ρ*) = exp(i * Q_j(I)) * ρ_j as a NormalFormSeries, truncated.

    Convention
    ----------
    2π Q_j = Σ_α ω_{j,α} I^α  →  f_j = exp(i * Q_j(I)) ρ_j
    (ω_{j,α} already includes 2π.)

    Parameters
    ----------
    plane_j : int
        Plane index (0..P-1) with total dimension dim = 2P.
    omega_j : dict[tuple[int], scalar]
        Detuning coefficients. May be:
          - Reduced form {(j1,j2,...): ω}
          - Full 2n form {(j1,j1,j2,j2,...): ω} as returned by symbolic_detuning().
    order : int
        Truncation order for exponential expansion.
    mp : str | module
        Backend ('numpy' or 'sympy').

    Returns
    -------
    pt.NormalFormSeries
        Truncated NormalFormSeries representing f_j(ρ,ρ*) = exp(i Q_j(I)) ρ_j.
    """

    # ------------------------------------------
    # Detect and normalize input dictionary
    # ------------------------------------------
    assert isinstance(omega_j, dict), "omega_j must be a dict of coefficients"

    first_key = next(iter(omega_j))
    key_len = len(first_key)

    # Detect if this is full 2n index (e.g. (j,j,k,k,...))
    if key_len % 2 == 0 and all(first_key[2*r] == first_key[2*r+1] for r in range(key_len // 2)):
        P = key_len // 2  # number of planes
        dim = key_len
        # Collapse full 2n indices → reduced nD indices
        omega_reduced = {}
        for k, v in omega_j.items():
            reduced = tuple(k[2*r] for r in range(P))
            omega_reduced[reduced] = v
        omega_j = omega_reduced
    else:
        # Reduced form already
        P = key_len
        dim = 2 * P

    # ------------------------------------------
    # Backend setup
    # ------------------------------------------
    base = pt.NormalFormSeries(dim=dim, max_order=order, mp=mp)
    mp = base.mp
    sym_mode = (mp.name == "sympy")

    if sym_mode:
        import sympy as sp
        ONE, ZERO, IUNIT = sp.Integer(1), sp.Integer(0), sp.I
        def sym(v): return v if isinstance(v, sp.Basic) else sp.sympify(v)
    else:
        ONE, ZERO, IUNIT = 1.0, 0.0, mp.I
        def sym(v): return v

    # ------------------------------------------
    # Build Q_j(ρ,ρ*) = Σ ω_{j,α} Π_r (ρ_r ρ_r*)^{α_r}
    # ------------------------------------------
    Q_coeffs = {}
    for alpha, w in omega_j.items():
        idx = [0] * dim
        for r, a_r in enumerate(alpha):
            if a_r:
                idx[2*r]   += a_r
                idx[2*r+1] += a_r
        t = tuple(idx)
        Q_coeffs[t] = Q_coeffs.get(t, ZERO) + sym(w)

    Qj = base.copy(coeff_dict=Q_coeffs)

    # ------------------------------------------
    # Separate constant term: Q = w₀ + Q̃
    # ------------------------------------------
    zero_idx = (0,) * dim
    w0 = Qj.coeffs.get(zero_idx, ZERO)
    if w0 != 0:
        Qtilde_coeffs = dict(Qj.coeffs)
        Qtilde_coeffs[zero_idx] = sym(Qtilde_coeffs.get(zero_idx, ZERO) - w0)
        if Qtilde_coeffs[zero_idx] == 0:
            Qtilde_coeffs.pop(zero_idx)
        Qtilde = base.copy(coeff_dict=Qtilde_coeffs)
    else:
        Qtilde = Qj

    scalar = mp.exp(IUNIT * w0) if w0 != 0 else ONE

    # ------------------------------------------
    # Expand exp(i Q̃) = Σ (i Q̃)^n / n!
    # ------------------------------------------
    if len(Qtilde.coeffs):
        A = IUNIT * Qtilde
        A_powers = mathlib.list_powers(A, order)

        if sym_mode:
            import sympy as sp
            coeffs = [sp.Rational(1, math.factorial(n)) for n in range(order + 1)]
        else:
            coeffs = [1.0 / math.factorial(n) for n in range(order + 1)]

        expA = mathlib.series_expansion(coeffs, A_powers)
    else:
        expA = base.copy(coeff_dict={zero_idx: ONE})

    # ------------------------------------------
    # Multiply by ρ_j and scalar
    # ------------------------------------------
    basis_idx = [0] * dim
    basis_idx[2 * plane_j] = 1
    rho_j = base.copy(coeff_dict={tuple(basis_idx): ONE})

    f_j = scalar * (expA * rho_j)

    # ------------------------------------------
    # Safe truncation
    # ------------------------------------------
    f_j = f_j.truncate(max_order=order)
    if sym_mode:
        f_j = f_j.nsimplify()

    return f_j


def _partial_of(series, plane_j, by = "rho"):
    """
    Algebraic partial derivative with respect to rho_j or rho_j*,
    acting on the (2*j, 2*j+1) index pair.
    """
    assert by in ("rho", "rho*")
    dim = series.dim
    P   = dim // 2
    assert 0 <= plane_j < P

    i_r  = 2 * plane_j     # index of rho_j exponent
    i_rs = i_r + 1         # index of rho_j* exponent

    out = {}
    if by == "rho":
        # d/d rho_j: (m, n) -> (m-1, n), value *= m
        for k, c in series.coeffs.items():
            m = k[i_r]
            if m > 0:
                newk = list(k)
                newk[i_r] = m - 1
                newk = tuple(newk)
                out[newk] = out.get(newk, 0) + m * c
    else:
        # d/d rho_j*: (m, n) -> (m, n-1), value *= n
        for k, c in series.coeffs.items():
            n = k[i_rs]
            if n > 0:
                newk = list(k)
                newk[i_rs] = n - 1
                newk = tuple(newk)
                out[newk] = out.get(newk, 0) + n * c

    return series.copy(coeff_dict=out)


