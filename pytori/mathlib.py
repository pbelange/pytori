import numpy as np


import pytori as pt


# Main Mathlibs
#=========================================================================================
def import_mathlib(mp):
    if mp == 'numpy':
        return MathlibNumpy
    elif mp == 'sympy':
        return MathlibSympy
    else:
        return mp  # assume already-imported module
#=========================================================================================


class MathlibSympy(object):
    name = "sympy"
    from sympy import sqrt, exp, sin, cos, pi, tan, conjugate, I, Matrix
    from sympy import Abs as abs
    from sympy import acos as arccos
    from sympy import asin as arcsin
    from sympy import atan as arctan
    from sympy import atan2 as arctan2
    from sympy import re as real
    from sympy import im as imag
    from sympy import Basic
    from sympy import latex as _latex
    from sympy import sympify, nsimplify, simplify, expand
    from IPython.display import display as _display, Latex as _Latex
    from sympy import pretty as pretty
    from sympy import sstr as sstr
    from sympy import Eq as Eq
    from sympy import arg as angle
    from sympy import nan

    @staticmethod
    def print_eq(lhs, rhs):
        result = r"${} \mapsto {}$".format(MathlibSympy._latex(lhs), MathlibSympy._latex(rhs))
        MathlibSympy._display(MathlibSympy._Latex(result))

    # function to wrap numpy zeros passing all arguments:
    @staticmethod
    def zeros(*args, **kw):
        return MathlibSympy.Matrix(np.zeros(*args, **kw))
    

class MathlibNumpy(object):
    name    = "numpy"
    I       = 1j
    from numpy import angle, sqrt, exp, sin, cos, abs, pi, tan, conjugate,arccos, arcsin, arctan,arctan2,real,imag,zeros
    from numpy import complex128 as Basic
    from numpy import nan


#======================================================================
# Global scalar types — unified across backends
#======================================================================
SCALARS = (
    int,
    float,
    complex,
    np.generic,          # covers numpy scalar types (np.float64, np.complex128, etc.)
    MathlibNumpy.Basic,  # numpy complex128 alias (already imported)
    MathlibSympy.Basic,  # sympy.Basic superclass (symbolic scalars)
)
#=========================================================================================




def list_powers(Psi, pwr):
    """
    Generate [Psi**0, Psi**1, ..., Psi**pwr].
    Works for numbers, NumPy arrays, SymPy expressions, or BaseSeries-like objects.

    Parameters
    ----------
    Psi : object
        Base quantity (scalar, array, symbolic, or series).
    pwr : int
        Highest power to compute (>= 0).

    Returns
    -------
    list
        List of successive powers [Psi**0, Psi**1, ..., Psi**pwr].
    """
    assert isinstance(pwr, int) and pwr >= 0
    result = Psi**0
    lst = [result]
    for _ in range(pwr):
        result = result * Psi
        lst.append(result)
    return lst


def series_expansion(coeffs, Psi):
    """
    Evaluate a truncated series f(Psi) = Σ_k coeffs[k] * Psi**k.

    Parameters
    ----------
    coeffs : sequence
        List or array of coefficients [c0, c1, ..., cN].
    Psi : object | list
        Base quantity to exponentiate or its precomputed powers.
        If Psi is a list, it is assumed to be [Psi**0, Psi**1, ...].

    Returns
    -------
    object
        Combined result of the expansion.
    """
    # Allow precomputed powers to save recomputation
    if isinstance(Psi, list):
        powers = Psi
    else:
        powers = list_powers(Psi, len(coeffs) - 1)

    assert len(powers) == len(coeffs), \
        f"Length mismatch: got {len(coeffs)} coeffs, {len(powers)} powers."

    # Ensure we get the right zero type (numeric, symbolic, or series)
    result = 0 * powers[0]
    for ck, Psik in zip(coeffs, powers):
        result += ck * Psik
    return result


def _tune_angle(z,mp='numpy'):
    """Extract angle between -0.5 and 0.5 from complex number z"""
    mp      = import_mathlib(mp)
    zn      = z/mp.abs(z)
    theta   = mp.arctan2(mp.imag(zn), mp.real(zn))  # θ ∈ (-π, π]
    q       = theta / (2.0 * mp.pi)                 # q ∈ (-0.5, 0.5]
    if mp.name == 'sympy':
        q = q.evalf()
    return q

# Taken from https://github.com/xsuite/xtrack/blob/main/ducktrack/elements.py
def _arrayofsize(ar, size):
    ar = np.array(ar)
    if len(ar) == 0:
        return np.zeros(size, dtype=ar.dtype)
    elif len(ar) < size:
        ar = np.hstack([ar, np.zeros(size - len(ar), dtype=ar.dtype)])
    return ar

# Extracting position and momentum series
#=========================================================================================
def Qj(Psi,bet0,mp='numpy'):
    """
    Calculate the position series for a given Psi.
    """
    mp      = import_mathlib(mp)
    factor  = mp.sqrt(bet0)/2
    return factor*(Psi.conjugate()+Psi)

def Pj(Psi,bet0,mp='numpy'):
    """
    Calculate the momentum series for a given Psi.
    """
    mp      = import_mathlib(mp)
    factor  = 1/(2*mp.I*mp.sqrt(bet0))
    return factor*(Psi.conjugate()-Psi)



def solve_system(eq_list, unknowns):
    import sympy as sp

    sol = sp.solve(eq_list, list(unknowns), dict=True)

    # --- Simplify each value before returning ---
    for i, s in enumerate(sol):
        for k, v in s.items():
            sol[i][k] = sp.simplify(v)
    return sol



#=========================================================================================
# Poincaré integral helper
#=========================================================================================


def _phi(A,n,int_angle,Tx=0,Ty=0,Tz=0,mp='numpy'):
    mp  = import_mathlib(mp)
    dim = len(n[0])
    if dim == 1:
        if int_angle != 'x':
            raise ValueError("For dim=1, int_angle must be 'x'.")
        phi = [mp.angle(Ak) for Ak in A]

    elif dim == 2:
        if int_angle == 'x':
            phi = [mp.angle(Ak) + nk[1]*Ty for Ak, nk in zip(A, n)]
        elif int_angle == 'y':
            phi = [mp.angle(Ak) + nk[0]*Tx for Ak, nk in zip(A, n)]
        else:
            raise ValueError(f"Invalid int_angle='{int_angle}' for dim=2")

    elif dim == 3:
        if int_angle == 'x':
            phi = [mp.angle(Ak) + nk[1]*Ty + nk[2]*Tz for Ak, nk in zip(A, n)]
        elif int_angle == 'y':
            phi = [mp.angle(Ak) + nk[0]*Tx + nk[2]*Tz for Ak, nk in zip(A, n)]
        elif int_angle == 'z':
            phi = [mp.angle(Ak) + nk[0]*Tx + nk[1]*Ty for Ak, nk in zip(A, n)]
        else:
            raise ValueError(f"Invalid int_angle='{int_angle}' for dim=3")

    else:
        raise ValueError(f"Invalid dimension: {dim}")

    return phi



def poincare_integral(A,n,int_angle,Tx=0,Ty=0,Tz=0,mp='numpy'):
    mp  = import_mathlib(mp)
    dim = len(n[0])

    phi  = _phi(A,n,int_angle,Tx,Ty,Tz,mp)
    jidx = {'x':0,'y':1,'z':2}[int_angle]
    Ajk  = mp.pi*sum(mp.abs(Ak)*mp.abs(Aj)*nk[jidx]*mp.cos(phik-phij)   for nk,Ak,phik in zip(n,A,phi) 
                                                                        for nj,Aj,phij in zip(n,A,phi) 
                                                                        if nj[jidx]==nk[jidx])
    
    return Ajk

def poincare_avg(A,n,int_angle,mp='numpy'):
    mp      = import_mathlib(mp)
    jidx    = {'x':0,'y':1,'z':2}[int_angle]
    return mp.pi * sum( nk[jidx]* mp.abs(Ak)**2  for nk,Ak in zip(n,A))




def gegenbauer_coeffs(N, alpha, x, mp='numpy'):
    mp = import_mathlib(mp)
    out = [None] * (N + 1)

    # Recurrence
    Ckm2 = mp.Basic(1)
    out[0] = +Ckm2
    if N == 0:
        return out

    Ckm1 = 2 * alpha * x
    out[1] = -Ckm1
    sign = -1
    for k in range(2, N + 1):
        kf = mp.Basic(k)
        Ck = (2 * (kf + alpha - 1) / kf) * x * Ckm1 - ((kf + 2*alpha - 2) / kf) * Ckm2
        sign = -sign
        out[k] = sign * Ck
        Ckm2, Ckm1 = Ckm1, Ck
    return out


def _truncated_gegenbauer_series(Pz_pwrs, alpha, beta_rel, mp='numpy'):
    """
    Construct the truncated Gegenbauer series C_z^{(alpha)} up to degree N.

    Parameters
    ----------
    Pz_pwrs : list
        Precomputed powers [u^0, u^1, ..., u^N] with u = β_rel * Pz.
    alpha : float
        Gegenbauer order (+½ or −½ for symplectic chromatic factors).
    beta_rel : float
        Relativistic β (scaling factor for u = β_rel * Pz).
    mp : str or module
        Mathlib backend ('numpy' or 'sympy').

    Returns
    -------
    Cz : same type as Pz
        Truncated series Σ c_k u^k with consistent backend typing.
    """
    mp = import_mathlib(mp)
    N = len(Pz_pwrs) - 1
    coeffs = gegenbauer_coeffs(N, alpha, 1/beta_rel, mp=mp)
    return series_expansion(coeffs, Pz_pwrs)


def _d_truncated_gegenbauer_series(Pz_pwrs, alpha, beta_rel, mp='numpy'):
    """
    Derivative of the truncated Gegenbauer series C_z^{(alpha)}(β_rel * Pz).

    Parameters
    ----------
    Pz_pwrs : list
        Powers [u^0, u^1, ..., u^N].
    alpha : float
        Gegenbauer order (+½ or −½).
    beta_rel : float
        Relativistic β.
    mp : str or module
        Mathlib backend.

    Returns
    -------
    dCz_dPz : same type as Pz
        Truncated derivative β_rel Σ_{k=1}^N k c_k u^{k−1}.
    """
    mp = import_mathlib(mp)
    N = len(Pz_pwrs) - 1
    if N <= 0:
        return 0 * Pz_pwrs[0]
    coeffs = gegenbauer_coeffs(N, alpha, 1/beta_rel, mp=mp)
    coeffs_d = [k * c for k, c in enumerate(coeffs)][1:]
    return beta_rel * series_expansion(coeffs_d, Pz_pwrs[:-1])



# Normalisation factors
#=========================================================================================
def W_to_lambda(W_matrix):
    """
    Extract lambda^+ and lambda^- from W_matrix of shape (2*dim, 2*dim).
    """
    W_matrix = np.asarray(W_matrix)
    assert W_matrix.shape[0] == W_matrix.shape[1], "W must be square"
    assert W_matrix.shape[0] % 2 == 0, "W must have even dimensions"

    dim = W_matrix.shape[0] // 2
    lambda_plus = np.zeros((dim, dim), dtype=complex)
    lambda_minus = np.zeros((dim, dim), dtype=complex)

    for i in range(dim):
        for j in range(dim):
            Oij = W_matrix[2*i:2*i+2, 2*j:2*j+2]
            a, b = Oij[0, 0], Oij[0, 1]
            c, d = Oij[1, 0], Oij[1, 1]
            lambda_plus[i, j]  = 0.5 * (a + d) - 0.5j * (c - b)
            lambda_minus[i, j] = 0.5 * (a - d) - 0.5j * (c + b)
    return lambda_plus, lambda_minus


def lambda_to_W(lambda_plus, lambda_minus):
    """
    Reconstruct W_matrix of shape (2*dim, 2*dim) from lambda^+ and lambda^-.
    """
    lambda_plus = np.asarray(lambda_plus)
    lambda_minus = np.asarray(lambda_minus)
    assert lambda_plus.shape == lambda_minus.shape
    dim = lambda_plus.shape[0]

    W = np.zeros((2*dim, 2*dim))
    for i in range(dim):
        for j in range(dim):
            lp, lm = lambda_plus[i, j], lambda_minus[i, j]
            a = np.real(lp + lm)
            d = np.real(lp - lm)
            c = -np.imag(lp + lm)
            b =  np.imag(lp - lm)
            W[2*i:2*i+2, 2*j:2*j+2] = [[a, b], [c, d]]
    return W


def co_geo_normalization(nemitt_x=None, nemitt_y=None, nemitt_z=None,
                         particle_on_co=None, beta_rel=None, gamma_rel=None):
    """
    Compute complex closed orbit vector and geometric emittances based on normalized emittances
    and particle reference coordinates.

    Parameters
    ----------
    nemitt_x, nemitt_y, nemitt_z : float or array-like, optional
        Normalized emittances for each plane.
    particle_on_co : dict or xtrack.Particles, optional
        Closed orbit particle. Must include keys like 'x', 'px', etc., or be an xtrack object.
    beta_rel, gamma_rel : float, optional
        Relativistic beta and gamma, used if `particle_on_co` is a dict or None.

    Returns
    -------
    co : ndarray of shape (3,)
        Complex closed orbit: [x - i*px, y - i*py, zeta - i*ptau/beta0]
    gemitt : ndarray of shape (3,)
        Geometric emittances for x, y, z planes.
    """
    # Default return: no inputs provided
    if all(arg is None for arg in [nemitt_x, nemitt_y, nemitt_z, particle_on_co, beta_rel, gamma_rel]):
        return np.zeros(3, dtype=complex), np.ones(3)

    # Prepare closed orbit dictionary
    if particle_on_co is not None and not isinstance(particle_on_co, dict):
        import xobjects as xo
        co_dict = particle_on_co.copy(_context=xo.context_default).to_dict()
        for key in ['x', 'px', 'y', 'py', 'zeta', 'ptau', 'beta0', 'gamma0']:
            val = co_dict[key]
            if np.ndim(val) > 0:
                co_dict[key] = val[0]
    else:
        co_dict = {
            'beta0': beta_rel or 0,
            'gamma0': gamma_rel or 0,
            'x': 0, 'px': 0,
            'y': 0, 'py': 0,
            'zeta': 0, 'ptau': 0
        }
        if particle_on_co is not None:
            co_dict.update(particle_on_co)

    # If any normalized emittance is provided, beta0 and gamma0 must be valid
    if any(e is not None for e in [nemitt_x, nemitt_y, nemitt_z]):
        assert co_dict['beta0'] > 0 and co_dict['gamma0'] > 0, "beta0 and gamma0 must be defined"

    # Compute geometric emittances
    def compute_geom_emit(nemitt):
        return 1.0 if nemitt is None else nemitt / co_dict['beta0'] / co_dict['gamma0']

    gemitt = np.array([
        compute_geom_emit(nemitt_x),
        compute_geom_emit(nemitt_y),
        compute_geom_emit(nemitt_z)
    ])

    co = np.array([
        co_dict['x'] - 1j * co_dict['px'],
        co_dict['y'] - 1j * co_dict['py'],
        co_dict['zeta'] - 1j * co_dict['ptau'] / co_dict['beta0']
    ], dtype=complex)

    return co, gemitt
#=========================================================================================

