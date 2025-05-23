import numpy as np


def Qj(Psi):
    """
    Calculate the position series for a given Psi.
    """
    factor = np.sqrt(Psi.bet0)/2
    return factor*(Psi.conj()+Psi)

def Pj(Psi):
    """
    Calculate the momentum series for a given Psi.
    """
    factor = 1/(2*1j*np.sqrt(Psi.bet0))
    return factor*(Psi.conj()-Psi)


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


def drift(Psix=None, Psiy=None, Psiz=None, ds=0,exp_order=20):
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
    Px = Pj(Psix) if Psix is not None else 0
    Py = Pj(Psiy) if Psiy is not None else 0
    Pz = Pj(Psiz) if Psiz is not None else 0

    if Psiz is not None:
        div_terms = list_powers(-1*Pz, exp_order)
    else:
        div_terms = [1] 

    div_sum = sum(div_terms)

    if Psix is not None:
        _Psix = Psix + ds / np.sqrt(Psix.bet0) * Px * div_sum
    if Psiy is not None:
        _Psiy = Psiy + ds / np.sqrt(Psiy.bet0) * Py * div_sum
    if Psiz is not None:
        div_sum2 = sum((k + 1) * term for k, term in enumerate(div_terms))
        _Psiz = Psiz - ds / np.sqrt(Psiz.bet0) * (Px**2 + Py**2) / 2 * div_sum2

    return _Psix, _Psiy, _Psiz






