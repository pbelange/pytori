import numpy as np
import pytori as pt

# Default math library, numpy
_mp = pt.algebra.MathlibNumpy


# Extracting position and momentum series
#=========================================================================================
def Qj(Psi,bet0,mp=_mp):
    """
    Calculate the position series for a given Psi.
    """
    factor = mp.sqrt(bet0)/2
    return factor*(Psi.conjugate()+Psi)

def Pj(Psi,bet0,mp=_mp):
    """
    Calculate the momentum series for a given Psi.
    """
    factor = 1/(2*1j*mp.sqrt(bet0))
    return factor*(Psi.conjugate()-Psi)
#=========================================================================================



# Utilities
#=========================================================================================
def list_powers(Psi,pwr):
    """
    Generate a list of powers of Psi to reduce number of operations.
    """
    assert isinstance(pwr, int) and pwr >= 0
    identity = Psi**0
    result   = identity
    lst      = [identity]
    for _ in range(pwr):
        result = result * Psi
        lst.append(result)
    return lst


# Taken from https://github.com/xsuite/xtrack/blob/main/ducktrack/elements.py
def _arrayofsize(ar, size):
    ar = np.array(ar)
    if len(ar) == 0:
        return np.zeros(size, dtype=ar.dtype)
    elif len(ar) < size:
        ar = np.hstack([ar, np.zeros(size - len(ar), dtype=ar.dtype)])
    return ar

def _init_beta0(Psix, Psiy, Psiz, bet0x, bet0y, bet0z):
    bet0x = bet0x if bet0x is not None else getattr(Psix, 'bet0', 1)
    bet0y = bet0y if bet0y is not None else getattr(Psiy, 'bet0', 1)
    bet0z = bet0z if bet0z is not None else getattr(Psiz, 'bet0', 1)
    return bet0x, bet0y, bet0z
#=========================================================================================



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



def drift(Psix=None, Psiy=None, Psiz=None, ds=0,exp_order=20,bet0x=None, bet0y=None, bet0z=None,mp=_mp):
    """
    Apply the drift transformation: 
    H = 1/2 * (px^2 + py^2) / (1 + pz^2)
    
    Parameters:
        ds                  : float — drift length
        Psix, Psiy, Psiz    : FourierSeriesND, np.array or None — projections of the Fourier series
        exp_order           : int — number of terms in the series expansion

    Returns:
        (_Psix, _Psiy, _Psiz): transformed Fourier series (or None if not provided)
    """
    _Psix, _Psiy, _Psiz = None, None, None
    # Ensure bet0 values are provided
    bet0x, bet0y, bet0z = _init_beta0(Psix, Psiy, Psiz, bet0x, bet0y, bet0z)

    # Extract coordinates
    Px = Pj(Psix,bet0x,mp=mp) if Psix is not None else 0
    Py = Pj(Psiy,bet0y,mp=mp) if Psiy is not None else 0
    Pz = Pj(Psiz,bet0z,mp=mp) if Psiz is not None else 0

    if Psiz is not None:
        div_terms = list_powers(-1*Pz, exp_order)
    else:
        div_terms = [1] 

    div_sum = sum(div_terms)

    if Psix is not None:
        _Psix = Psix + ds / mp.sqrt(bet0x) * Px * div_sum
    if Psiy is not None:
        _Psiy = Psiy + ds / mp.sqrt(bet0y) * Py * div_sum
    if Psiz is not None:
        div_sum2 = sum((k + 1) * term for k, term in enumerate(div_terms))
        _Psiz = Psiz - ds / mp.sqrt(bet0z) * (Px**2 + Py**2) / 2 * div_sum2

    return _Psix, _Psiy, _Psiz




def multipole(Psix=None, Psiy=None, Psiz=None, knl=[],ksl=[],bet0x=None, bet0y=None, bet0z=None,mp=_mp):
    """
    Apply the thin-multipole transformation: 
    H = Re[(knl + i * ksl) * (x + i * y)^(n+1) / (n+1)!]

    Adapted from https://github.com/xsuite/xtrack/blob/main/ducktrack/elements.py
    Horner-like recursion to avoid full expansion
    
    Parameters:
        knl                 : list - Normalized integrated strength of normal components
        ksl                 : list - Normalized integrated strength of skew components
        Psix, Psiy, Psiz    : FourierSeriesND, np.array or None — projections of the Fourier series

    Returns:
        (_Psix, _Psiy, _Psiz): transformed Fourier series (or None if not provided)
    """

    # Initialize
    order = max(len(knl), len(ksl)) - 1
    knl = _arrayofsize(knl, order + 1)
    ksl = _arrayofsize(ksl, order + 1)

    # Extracting position series
    _Psix, _Psiy, _Psiz = None, None, None
    # Ensure bet0 values are provided
    bet0x, bet0y, bet0z = _init_beta0(Psix, Psiy, Psiz, bet0x, bet0y, bet0z)
    X = Qj(Psix,bet0x,mp=mp) if Psix is not None else 0
    Y = Qj(Psiy,bet0y,mp=mp) if Psiy is not None else 0
    Z = Qj(Psiz,bet0z,mp=mp) if Psiz is not None else 0


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


    if Psix is not None:
        _Psix = Psix - 1j * mp.sqrt(bet0x) * dpx
    if Psiy is not None:
        _Psiy = Psiy - 1j * mp.sqrt(bet0y) * dpy
    if Psiz is not None:
        _Psiz = Psiz

    return _Psix, _Psiy, _Psiz



def phys2norm(Psix=None, Psiy=None, Psiz=None, lambda_plus=None, lambda_minus=None, W_matrix=None,
              nemitt_x=None, nemitt_y=None, nemitt_z=None, particle_on_co=None,beta_rel = None,gamma_rel = None):
    """
    Apply normalization transformation to coupled phase space variables (ψ_x, ψ_y, ψ_ζ),
    converting them into decoupled (normalized) variables (ψ̃_x, ψ̃_y, ψ̃_ζ).

    Parameters
    ----------
    Psix, Psiy, Psiz : FourierSeriesND, complex, or compatible objects, optional
        Coupled (physical) phase space projections in x, y, and zeta. 
        Only the provided dimensions will be used, and the rest are assumed absent.

    lambda_plus : ndarray of shape (dim, dim), optional
        Matrix of λ⁺ optical functions. Must be provided if W_matrix is not.

    lambda_minus : ndarray of shape (dim, dim), optional
        Matrix of λ⁻ optical functions. Must be provided if W_matrix is not.

    W_matrix : ndarray of shape (2*dim, 2*dim), optional
        Denormalization matrix W. If provided, it will be converted into λ⁺ and λ⁻.

    Returns
    -------
    Psix_tilde, Psiy_tilde, Psiz_tilde : tuple
        Normalized (decoupled) phase space variables.
        Returned in a 3-element tuple; unused dimensions are returned as None.

    """
    psi_list = [Psix, Psiy, Psiz]
    psi_vec = [psi for psi in psi_list if psi is not None]
    active_dims = [i for i, psi in enumerate(psi_list) if psi is not None]
    dim = len(psi_vec)

    if W_matrix is not None:
        assert lambda_plus is None and lambda_minus is None, "Provide either W_matrix or lambda matrices, not both."
        lambda_plus, lambda_minus = W_to_lambda(W_matrix)
    else:
        assert lambda_plus is not None and lambda_minus is not None, "lambda_plus and lambda_minus must be provided."

    assert lambda_plus.shape == (dim, dim), f"Expected lambda matrices of shape ({dim},{dim}), got {lambda_plus.shape}"

    # Closed orbit substraction
    #========================================================
    co, geo = co_geo_normalization(nemitt_x=nemitt_x, nemitt_y=nemitt_y, nemitt_z=nemitt_z,particle_on_co=particle_on_co, beta_rel=beta_rel, gamma_rel=gamma_rel)        
    for i, idx in enumerate(active_dims):
        psi_vec[idx] = psi_vec[idx] - co[idx]
    #========================================================

    # Normalization transformation
    #========================================================
    psi_tilde = [0] * dim
    for i in range(dim):
        for j in range(dim):
            psi_tilde[i] += np.conj(lambda_plus[j, i]) * psi_vec[j] - lambda_minus[j, i] * psi_vec[j].conjugate()

    result = [None, None, None]
    for i, idx in enumerate(active_dims):
        result[idx] = psi_tilde[i]
    #=========================================================

    # Emittance rescaling
    #=========================================================
    for i, idx in enumerate(active_dims):
        assert result[idx] is not None, f"Expected result[{idx}] to be set for normalization"
        result[idx] = result[idx] / np.sqrt(geo[idx])
    #=========================================================

    return tuple(result)



def norm2phys(Psix=None, Psiy=None, Psiz=None, lambda_plus=None, lambda_minus=None, W_matrix=None,
              nemitt_x=None, nemitt_y=None, nemitt_z=None, particle_on_co=None,beta_rel = None,gamma_rel = None):
    """
    Apply inverse normalization transformation to decoupled phase space variables (ψ̃_x, ψ̃_y, ψ̃_ζ),
    reconstructing the coupled (physical) variables (ψ_x, ψ_y, ψ_ζ).

    Parameters
    ----------
    Psix, Psiy, Psiz : FourierSeriesND, complex, or compatible objects, optional
        Normalized (decoupled) phase space projections in x, y, and zeta.
        Only the provided dimensions will be used, and the rest are assumed absent.

    lambda_plus : ndarray of shape (dim, dim), optional
        Matrix of λ⁺ optical functions. Must be provided if W_matrix is not.

    lambda_minus : ndarray of shape (dim, dim), optional
        Matrix of λ⁻ optical functions. Must be provided if W_matrix is not.

    W_matrix : ndarray of shape (2*dim, 2*dim), optional
        Denormalization matrix W. If provided, it will be converted into λ⁺ and λ⁻.

    Returns
    -------
    Psix_phys, Psiy_phys, Psiz_phys : tuple
        Coupled (physical) phase space variables.
        Returned in a 3-element tuple; unused dimensions are returned as None.

    """
    psi_list = [Psix, Psiy, Psiz]
    psi_vec = [psi for psi in psi_list if psi is not None]
    active_dims = [i for i, psi in enumerate(psi_list) if psi is not None]
    dim = len(psi_vec)

    if W_matrix is not None:
        assert lambda_plus is None and lambda_minus is None, "Provide either W_matrix or lambda matrices, not both."
        lambda_plus, lambda_minus = W_to_lambda(W_matrix)
    else:
        assert lambda_plus is not None and lambda_minus is not None, "lambda_plus and lambda_minus must be provided."

    assert lambda_plus.shape == (dim, dim), f"Expected lambda matrices of shape ({dim},{dim}), got {lambda_plus.shape}"


    # Emittance rescaling
    #=========================================================
    co, geo = co_geo_normalization(nemitt_x=nemitt_x, nemitt_y=nemitt_y, nemitt_z=nemitt_z,particle_on_co=particle_on_co, beta_rel=beta_rel, gamma_rel=gamma_rel)        
    for i, idx in enumerate(active_dims):        
        psi_vec[idx] = psi_vec[idx] * np.sqrt(geo[idx])
    #=========================================================


    # DE-normalization transformation
    #========================================================
    psi_phys = [0] * dim
    for i in range(dim):
        for j in range(dim):
            psi_phys[i] += lambda_plus[i, j] * psi_vec[j] + lambda_minus[i, j] * psi_vec[j].conjugate()

    result = [None, None, None]
    for i, idx in enumerate(active_dims):
        result[idx] = psi_phys[i]
    #========================================================

    # Closed orbit correction
    #========================================================     
    for i, idx in enumerate(active_dims):
        assert result[idx] is not None, f"Expected result[{idx}] to be set for normalization"
        result[idx] = result[idx] + co[idx]
    #========================================================

    return tuple(result)
