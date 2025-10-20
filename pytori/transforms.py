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


def _truncated_gegenbauer_series(Pz_pwrs, alpha, beta_rel, mp=_mp):
    """
    Construct the truncated chromatic factor Cz (or its inverse) up to degree N.

    Definition
    ----------
    Cz(u) = sum_{k=0}^∞ c_k u^k ,    with u = beta_rel * Pz
    where the coefficients c_k depend on (alpha, 1/beta_rel).
    """
    
    N      = len(Pz_pwrs) - 1  # Degree of truncation
    coeffs = gegenbauer_coeffs(N, alpha, 1/beta_rel)  # c_0..c_N

    return sum(c * p for c, p in zip(coeffs, Pz_pwrs))


def _d_truncated_gegenbauer_series(Pz_pwrs, alpha, beta_rel, mp=_mp):
    """
    Build T_{N-1}[ d/dPz C_alpha(beta_rel*Pz) ] using the same Pz_pwrs.
    Pz_pwrs = [u^0,...,u^N]; derivative uses [u^0,...,u^{N-1}] and multiplies by beta_rel.
    """
    N = len(Pz_pwrs) - 1
    if N <= 0:
        return 0*Pz_pwrs[0]  # Derivative of constant is zero
    coeffs = gegenbauer_coeffs(N, alpha, 1/beta_rel)  # c_0..c_N
    # d/dPz sum_{k=0}^N c_k u^k = beta_rel * sum_{k=1}^N k c_k u^{k-1}
    u_pwrs = Pz_pwrs[:-1]  # [u^0,...,u^{N-1}]
    out = 0
    for k in range(1, N+1):
        out += (k * coeffs[k]) * u_pwrs[k-1]
    return beta_rel * out

#=========================================================================================



# Utilities
#=========================================================================================
def list_powers(Psi, pwr):
    """
    Generate [Psi**0, Psi**1, ..., Psi**pwr].
    Works for numbers, NumPy arrays, or SymPy expressions.
    """
    assert isinstance(pwr, int) and pwr >= 0
    result = Psi**0
    lst = [result]
    for _ in range(pwr):
        result = result * Psi
        lst.append(result)
    return lst

def gegenbauer_coeffs(N, alpha, x, dtype=float):
    """
    Return the coefficients a_k = (-1)^k * G_k^{(alpha)}(x) for k = 0..N (inclusive).
    That is, the output has length N+1 and represents the truncated series:
        sum_{k=0}^N a_k * u^k,   with u = (beta_rel * Pz)

    Returns
    -------
    out : np.ndarray shape (N+1,)
        Coefficients [a_0, a_1, ..., a_N].
    """
    assert isinstance(N, int) and N >= 0, "N must be an integer >= 0"

    x = dtype(x)
    out = np.empty(N + 1, dtype=dtype)

    # k = 0
    Ckm2 = dtype(1)         # G_0^{(alpha)}(x) = 1
    out[0] = Ckm2           # (-1)^0 * C0

    if N == 0:
        return out

    # k = 1
    Ckm1 = dtype(2) * dtype(alpha) * x  # G_1^{(alpha)}(x) = 2 alpha x
    out[1] = -Ckm1                      # (-1)^1 * C1

    sign = -1  # tracks (-1)^k; currently at k=1
    for k in range(2, N + 1):
        kf = dtype(k)
        Ck = (dtype(2) * (kf + dtype(alpha) - dtype(1)) / kf) * x * Ckm1 \
             - ((kf + dtype(2)*dtype(alpha) - dtype(2)) / kf) * Ckm2
        sign = -sign
        out[k] = sign * Ck
        Ckm2, Ckm1 = Ckm1, Ck

    return out

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



def drift(Psix=None, Psiy=None, Psiz=None, ds=0,particle_on_co=None,beta_rel = None,order=20,bet0x=None, bet0y=None, bet0z=None,mp=_mp):
    """
    Apply the drift transformation: 
    H = pz - delta + (px^2 + py^2) / 2(1 + delta)
    
    Parameters:
        ds                  : float — drift length
        Psix, Psiy, Psiz    : FourierSeriesND, np.array or None — projections of the Fourier series
        order               : int — truncation order for chromatic factor

    Returns:
        (_Psix, _Psiy, _Psiz): transformed Fourier series (or None if not provided)
    """
    _Psix, _Psiy, _Psiz = None, None, None

    # Ensure bet0 values are provided
    bet0x, bet0y, bet0z = _init_beta0(Psix, Psiy, Psiz, bet0x, bet0y, bet0z)

    if Psiz is not None:
        if beta_rel is None and particle_on_co is not None:
            beta_rel = particle_on_co.beta0[0] if hasattr(particle_on_co, 'beta0') else particle_on_co['beta0']
        assert beta_rel is not None, "beta_rel or particle_on_co with valid beta0 must be provided"

    # Extract canonical momenta
    Px = Pj(Psix, bet0x, mp=mp) if Psix is not None else 0
    Py = Pj(Psiy, bet0y, mp=mp) if Psiy is not None else 0
    Pz = Pj(Psiz, bet0z, mp=mp) if Psiz is not None else 0

    # Build chromatic factors truncated on the Hamiltonian (order N in Pz)
    if Psiz is not None:
        #---------------
        Pz_pwrs     = list_powers(beta_rel * Pz, pwr=order)  # [u^0,...,u^N]
        #---------------
        Cz          = _truncated_gegenbauer_series(Pz_pwrs, +0.5, beta_rel, mp=mp)
        dCz_dPz     = _d_truncated_gegenbauer_series(Pz_pwrs, +0.5, beta_rel, mp=mp)
        dCzinv_dPz  = _d_truncated_gegenbauer_series(Pz_pwrs, -0.5, beta_rel, mp=mp)
    else:
        Cz, Czinv, dCz_dPz, dCzinv_dPz = 1, 1, 0, 0

    # Apply the (exact-for-this-H) Lie map with consistent truncations
    if Psix is not None:
        _Psix = Psix + ds / mp.sqrt(bet0x) * (Px * Cz)

    if Psiy is not None:
        _Psiy = Psiy + ds / mp.sqrt(bet0y) * (Py * Cz)

    if Psiz is not None:
        Px2Py2_over2 = (Px*Px + Py*Py) / 2
        increment = 1 - dCzinv_dPz + dCz_dPz * Px2Py2_over2
        _Psiz = Psiz + ds / mp.sqrt(bet0z) * increment

    return _Psix, _Psiy, _Psiz






def bend(Psix=None, Psiy=None, Psiz=None, k0=None, h=None, particle_on_co=None,beta_rel = None,order=20,bet0x=None, bet0y=None, bet0z=None,mp=_mp):
    """
    Apply the thin dipole transformation: 
    H = -h*x*(1+delta) + k0*(x + h*x^2/2)
    
    Parameters:
        k0                  : normalized field strength
        Psix, Psiy, Psiz    : FourierSeriesND, np.array or None — projections of the Fourier series
        order               : int — truncation order for chromatic factor

    Returns:
        (_Psix, _Psiy, _Psiz): transformed Fourier series (or None if not provided)
    """
    _Psix, _Psiy, _Psiz = None, None, None
    # Ensure bet0 values are provided
    bet0x, bet0y, bet0z = _init_beta0(Psix, Psiy, Psiz, bet0x, bet0y, bet0z)
    
    if Psiz is not None:
        if beta_rel is None and particle_on_co is not None:
            beta_rel = particle_on_co.beta0[0] if hasattr(particle_on_co, 'beta0') else particle_on_co['beta0']
        assert beta_rel is not None, "beta_rel or particle_on_co with valid beta0 must be provided"

    if h is None and k0 is not None:
        h = k0
    elif h is not None and k0 is None:
        k0 = h
    elif h is None and k0 is None:
        raise ValueError("Either k0 or h must be provided")

    # Extract coordinates
    X   = Qj(Psix,bet0x,mp=mp) if Psix is not None else 0
    Pz  = Pj(Psiz,bet0z,mp=mp) if Psiz is not None else 0

    # Build chromatic factors truncated on the Hamiltonian (order N in Pz)
    if Psiz is not None:
        #---------------
        Pz_pwrs     = list_powers(beta_rel * Pz, pwr=order)  # [u^0,...,u^N]
        #---------------
        Czinv       = _truncated_gegenbauer_series(Pz_pwrs, -0.5, beta_rel, mp=mp)
        dCzinv_dPz  = _d_truncated_gegenbauer_series(Pz_pwrs, -0.5, beta_rel, mp=mp)
    else:
        Cz, Czinv, dCz_dPz, dCzinv_dPz = 1, 1, 0, 0


    if Psix is not None:
        _Psix = Psix + 1j * mp.sqrt(bet0x) * (k0*(1+h*X) - h*Czinv)
    if Psiy is not None:
        _Psiy = Psiy 
    if Psiz is not None:
        _Psiz = Psiz - 1 / mp.sqrt(bet0z) * dCzinv_dPz * h * X

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
              nemitt_x=None, nemitt_y=None, nemitt_z=None, particle_on_co=None,beta_rel = None,gamma_rel = None,mp=_mp):
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
            psi_tilde[i] += lambda_plus[j, i].conjugate() * psi_vec[j] - lambda_minus[j, i] * psi_vec[j].conjugate()

    result = [None, None, None]
    for i, idx in enumerate(active_dims):
        result[idx] = psi_tilde[i]
    #=========================================================

    # Emittance rescaling
    #=========================================================
    for i, idx in enumerate(active_dims):
        assert result[idx] is not None, f"Expected result[{idx}] to be set for normalization"
        result[idx] = result[idx] / mp.sqrt(geo[idx])
    #=========================================================

    return tuple(result)



def norm2phys(Psix=None, Psiy=None, Psiz=None, lambda_plus=None, lambda_minus=None, W_matrix=None,
              nemitt_x=None, nemitt_y=None, nemitt_z=None, particle_on_co=None,beta_rel = None,gamma_rel = None,mp=_mp):
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
        psi_vec[idx] = psi_vec[idx] * mp.sqrt(geo[idx])
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



def linear_map(Psix=None, Psiy=None, Psiz=None,
               Qvec=None, lambda_plus=None, lambda_minus=None, W_matrix=None,
               Lp_list=None,Lm_list=None,W_list=None,
               U_matrix=None, V_matrix=None, mp=_mp):
    """
    Apply a linear map to (ψ_x, ψ_y, ψ_ζ) in the complex basis Ψ, via
        Ψ' = U Ψ + V Ψ*
    where, if U,V are not provided:
        U =   Lp E Lp^† - Lm E^* Lm^†
        V = - Lp E Lm^T + Lm E^* Lp^T)
    with E = diag(exp(i*2π Q_j)).
    Arithmetic uses explicit loops to support NumPy / SymPy / custom backends.
    """

    # Active inputs (keep x,y,z ordering)
    psi_list    = [Psix, Psiy, Psiz]
    active_dims = [i for i, psi in enumerate(psi_list) if psi is not None]
    psi_vec     = [psi_list[i] for i in active_dims]
    dim = len(psi_vec)

    # Decide whether we build U,V or use provided
    need_UV = (U_matrix is None and V_matrix is None)

    if need_UV:
        # We need Qvec only if we're building U,V
        assert Qvec is not None, "Qvec (phase advances) must be provided when U,V are not."
        assert len(Qvec) >= dim, f"Expected at least {dim} phase advances"

        U,V = pt.linear_normal_form.construct_UV(Qvec=Qvec, lambda_plus=lambda_plus, lambda_minus=lambda_minus, W_matrix=W_matrix,
                                                 Lp_list=Lp_list,Lm_list=Lm_list,W_list=W_list, mp=mp)

    else:
        # Use provided U,V; coerce to list-of-lists for uniform indexing
        assert (U_matrix is not None) and (V_matrix is not None), \
            "Both U_matrix and V_matrix must be provided"
        try:
            U = U_matrix.tolist()
            V = V_matrix.tolist()
        except Exception:
            U = U_matrix
            V = V_matrix
        # Basic shape checks
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

    # Repack into (Psix, Psiy, Psiz)
    result = [None, None, None]
    for loc, idx in enumerate(active_dims):
        result[idx] = psi_out[loc]
    return tuple(result)