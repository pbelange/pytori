import numpy as np
from collections import defaultdict
import pandas as pd



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
    from sympy import sympify, nsimplify
    from IPython.display import display as _display, Latex as _Latex

    @staticmethod
    def print_eq(lhs, rhs):
        result = "${} \mapsto {}$".format(MathlibSympy._latex(lhs), MathlibSympy._latex(rhs))
        MathlibSympy._display(MathlibSympy._Latex(result))

    # function to wrap numpy zeros passing all arguments:
    @staticmethod
    def zeros(*args, **kw):
        return MathlibSympy.Matrix(np.zeros(*args, **kw))
    

class MathlibNumpy(object):
    name    = "numpy"
    I       = 1j
    from numpy import sqrt, exp, sin, cos, abs, pi, tan, conjugate,arccos, arcsin, arctan,arctan2,real,imag,zeros
    from numpy import complex128 as Basic
    

    


_mp = MathlibNumpy  # Default math library, can be changed to MathlibSympy for symbolic computations

class FourierSeriesND:
    def __init__(self, coeff_dict, dim = None,max_order=None, max_k=500, content_fraction=None, numerical_tol=1e-21,bet0=1,mp=_mp,complex_basis = True):
        """
        Initialize a FourierSeriesND object using sparse representation.

        Parameters:
            coeff_dict (dict): Mapping from index tuples (n1, n2, ..., nd) to complex values.
            max_order (tuple, optional): Truncate to modes where |n_i| <= max_order[i].
            max_k (int): Keep only the max_k largest coefficients by magnitude.
            content_fraction (float): Fraction of total energy to retain.            
            numerical_tol (float): Threshold for filtering near-zero values.
            bet0 (float): rescaling length for the Fourier series (psi has units of sqrt(m)!).
        """

        if dim is not None:
            self.dim = dim
        else:
            assert len(coeff_dict) > 0, "Cannot infer dimension from empty coeff_dict; please specify dim."
            self.dim = len(next(iter(coeff_dict))) 

        self.max_order = max_order
        self.max_k = max_k
        self.content_fraction = content_fraction
        self.numerical_tol = numerical_tol
        self.bet0 = bet0
        self.complex_basis = complex_basis
        if isinstance(mp, str):
            if mp == 'numpy':
                self.mp = MathlibNumpy
            elif mp == 'sympy':
                self.mp = MathlibSympy
                self.numerical_tol = None
            else:
                raise ValueError("mp must be 'numpy' or 'sympy'")
        else:
            self.mp = mp
            if self.mp is MathlibSympy:
                self.numerical_tol = None
        

        if max_order is not None or max_k is not None:
            coeff_dict = self._truncate_dict(coeff_dict, max_order, max_k,content_fraction)

        self._coeff_dict = {k: v for k, v in coeff_dict.items() if v != 0}

    

    def copy(self,coeff_dict=None):
        """Return a new copy of the Fourier series."""
        if coeff_dict is None:
            coeff_dict = {k: v for k, v in self._coeff_dict.items()}
        return FourierSeriesND(coeff_dict,dim = self.dim,max_order=self.max_order, max_k=self.max_k, content_fraction=self.content_fraction, 
                               numerical_tol=self.numerical_tol,bet0=self.bet0,mp=self.mp,complex_basis =self.complex_basis)

    def _truncate_dict(self, coeff_dict, max_order=None, max_k=None, content_fraction=None):
        """
        Truncate the Fourier series.

        Parameters:
            max_order (tuple): Max absolute value of each mode component.
            max_k (int): Keep only the max_k largest coefficients by magnitude.
        """
        if max_order:
            if isinstance(max_order, int):
                # max_order = (max_order,) * self.dim
                coeff_dict = {
                    k: v for k, v in coeff_dict.items()
                    if sum(np.abs(k)) <= max_order
                }
            elif isinstance(max_order, (list, tuple)):
                assert len(max_order) == self.dim, "max_order length must match dimension"
                coeff_dict = {
                k: v for k, v in coeff_dict.items()
                if all(abs(k[i]) <= max_order[i] for i in range(len(max_order)))
                }
            
        if max_k is not None and len(coeff_dict) > max_k:
            coeff_items = sorted(coeff_dict.items(), key=lambda kv: -abs(kv[1]))
            coeff_dict = dict(coeff_items[:max_k])

        if content_fraction is not None and 0 < content_fraction < 1:
            coeff_items = sorted(coeff_dict.items(), key=lambda kv: -abs(kv[1])**2)
            total_energy = sum(abs(v)**2 for _, v in coeff_items)
            cumulative_energy = 0
            cutoff_idx = 0
            for k, v in coeff_items:
                cumulative_energy += abs(v)**2
                cutoff_idx += 1
                if cumulative_energy / total_energy >= content_fraction:
                    break
            coeff_dict = dict(coeff_items[:cutoff_idx])
        return coeff_dict

    def to_dict(self, tol=None):
        """Convert internal storage to a dictionary of nonzero coefficients."""
        if tol is None:
            tol = self.numerical_tol
        if tol is not None:
            return {k: v for k, v in self._coeff_dict.items() if abs(v) > tol}
        else:
            return self._coeff_dict

    def coeffs(self, tol=None):
        """Return active coefficients as a sorted pandas DataFrame."""
        if tol is None:
            tol = self.numerical_tol
        coeff_dict = self.to_dict(tol)
        data = [(k, v) for k, v in coeff_dict.items()]
        df = pd.DataFrame(data, columns=["mode", "coefficient"])
        # return df.sort_values(by=df["coefficient"].apply(abs), ascending=False)
        try:
            return df.sort_values(by="coefficient", key=lambda col: abs(col), ascending=False).reset_index(drop=True)
        except:
            return df


    def truncate(self, max_order=None, max_k=None,content_fraction=None):
        """
        Returns a new FourierSeriesND with truncated coefficients.
        """
        return FourierSeriesND(self._coeff_dict, dim=self.dim, max_order=max_order, max_k=max_k,content_fraction=content_fraction, numerical_tol=self.numerical_tol,mp=self.mp,complex_basis=self.complex_basis)

    def dephase(self, phases):
        """
        Dephase the Fourier series by shifting the phases of each dimension.

        Parameters:
            phases (list or array): Phase shifts for each dimension in radians.
        """
        if isinstance(phases, (int, float, complex)):
            phases = [phases] * self.dim
        assert len(phases) == self.dim, "Length of phases must match dimension"
        assert self.complex_basis, "Dephasing is only applicable for complex basis Fourier series."
        phased_dict = {}
        for k, v in self._coeff_dict.items():
            phase_shift = sum(k[i] * phases[i] for i in range(self.dim))
            phased_dict[k] = v * self.mp.exp(self.mp.I * phase_shift)
        return self.copy(phased_dict)

    def star(self):
        """Return the complex conjugate series with reversed mode indices."""
        coeff_dict = self.to_dict()
        if self.complex_basis:
            conjugated = {tuple(-np.array(k)): self.mp.conjugate(v) for k, v in coeff_dict.items()}
        else:
            # Here we should flip the indices k[1],k[0],k[3],k[2], and so on
            conjugated = {}
            for k, v in coeff_dict.items():
                k_array = np.array(k)
                swapped_indices = []
                # Swap each consecutive pair
                for i in range(0, len(k_array), 2):
                    swapped_indices.extend([k_array[i+1], k_array[i]])
                swapped_k = tuple(swapped_indices)
                conjugated[swapped_k] = self.mp.conjugate(v)

        return self.copy(conjugated)

    def conj(self):
        """Alias for .star(), returns complex conjugate."""
        return self.star()
    
    def conjugate(self):
        """Alias for .star(), returns complex conjugate."""
        return self.star()

    def symsub(self, subs):
        """Substitute symbolic variables in the coefficients."""
        coeff_dict = self.to_dict()
        substituted = {k: v.subs(subs) if hasattr(v, 'subs') else v for k, v in coeff_dict.items()}
        return self.copy(substituted)
    
    def symevalf(self):
        """Numerical evaluation of symbolic coefficients."""
        coeff_dict = self.to_dict()
        substituted = {k: complex(v.evalf()) if hasattr(v, 'evalf') else v for k, v in coeff_dict.items()}
        return self.copy(substituted)
    
    def symsimplify(self):
        coeff_dict = self.to_dict()
        substituted = {k: MathlibSympy.nsimplify(MathlibSympy.sympify(v.simplify()), rational=True) if hasattr(v, 'simplify') else v for k, v in coeff_dict.items()}
        return self.copy(substituted)


    def symplectic_condition(self,order,subs={}):
        """
        Compute the symplectic condition:
            dPsi/dρ * dPsi*/dρ* - dPsi/dρ* * dPsi*/dρ = 1
        Returns a SymPy expression that should equal 1.
        """
        assert not self.complex_basis, "only for rho rho^* basis"
        import sympy as sp

        _r, _rs = sp.symbols('r r^*', real=True,positive=True)
        rho     = sp.symbols('rho', complex=True)
        rho_s   = sp.conjugate(rho)

        # reconstruct Psi(ρ, ρ*)
        Psi     = sum(c * rho**n[0] * rho_s**n[1] for n, c in self._coeff_dict.items())
        subs_in  = {rho:_r, rho_s:_rs}
        subs_out = {_r:rho, _rs:rho_s}
        dP_dr   = sp.diff(Psi.subs(subs_in), _r)
        dP_drs  = sp.diff(Psi.subs(subs_in), _rs)
        dPs_dr  = sp.diff(sp.conjugate(Psi).subs(subs_in), _r)
        dPs_drs = sp.diff(sp.conjugate(Psi).subs(subs_in), _rs)

        # subs_out = {_r:rho, _rs:rho_s}
        J = dP_dr * dPs_drs - dPs_dr * dP_drs
        J = J.simplify().expand()#.collect([_r,_rs,_r*_rs])

        # Truncate
        eps = sp.symbols('epsilon',real=True,positive=True)
        if order <= 1:
            order = 2
        J = J.subs({_r:eps*_r, _rs:eps*_rs}) + sp.O(eps**(order))
        J = J.expand().removeO().subs({eps:1})#.subs(subs_out)

        poly = sp.Poly(sp.expand((J-1)), _r, _rs)
        equations = []
        for (m, k), coeff in poly.terms():
            equations.append(sp.Eq(coeff.subs(subs).simplify(), 0))
        return equations
        
    def collapse(self,I):

        assert not self.complex_basis, "collapse only for rho rho^* basis"

        # Normalize input I
        if isinstance(I, (int, float, complex, self.mp.Basic)):
            assert self.dim == 2, "Scalar I only valid for 1D (2 complex coords)"
            I = [I]

        nI = self.dim // 2
        assert len(I) == nI, f"Expected {nI} action values, got {len(I)}"

        # Extract actions with defaults
        Ix = I[0] if nI >= 1 else 0
        Iy = I[1] if nI >= 2 else None
        Iz = I[2] if nI >= 3 else None

        # Initialize collapsed series
        base = self.copy()
        base.dim = nI
        base.complex_basis = True
        # Overwriting
        base = base.copy({})
        flat = base.copy({})

        for key, val in self._coeff_dict.items():
            # Pad key to full 6-tuple
            j, k, l, m, p, q = list(key) + [0] * (6 - len(key))

            # Spectral index (truncate to dimensionality)
            n_vec = (j - k, l - m, p - q)[:nI]

            # Construct amplitude term
            term = val
            if nI >= 1:
                term *= (2 * Ix) ** ((j + k) / 2)
            if nI >= 2:
                term *= (2 * Iy) ** ((l + m) / 2)
            if nI >= 3:
                term *= (2 * Iz) ** ((p + q) / 2)

            # Accumulate into flattened series
            flat += base.copy({n_vec: term})

        return flat

    def __add__(self, other):
        """Add two FourierSeriesND instances."""
        if isinstance(other, FourierSeriesND):
            dict1 = self.to_dict()
            dict2 = other.to_dict()
            all_keys = set(dict1.keys()) | set(dict2.keys())
            result_dict = {k: dict1.get(k, 0) + dict2.get(k, 0) for k in all_keys}
            max_order = self.max_order or other.max_order
            max_k = min(self.max_k, other.max_k) if self.max_k and other.max_k else self.max_k or other.max_k
            numerical_tol = min(self.numerical_tol, other.numerical_tol) if self.numerical_tol is not None else None
            return FourierSeriesND(result_dict,dim=self.dim, max_order=max_order, max_k=max_k, numerical_tol=numerical_tol,mp=self.mp,complex_basis=self.complex_basis)
        elif isinstance(other, (int, float, complex)):
            dict1 = self.to_dict()
            result_dict = dict(dict1)
            zero_mode = (0,) * self.dim
            result_dict[zero_mode] = result_dict.get(zero_mode, 0) + complex(other)
            return FourierSeriesND(result_dict, dim=self.dim, max_order=self.max_order, max_k=self.max_k, numerical_tol=self.numerical_tol,mp=self.mp,complex_basis=self.complex_basis)
        else:
            return NotImplemented

    def _sparse_convolve(self, dict1, dict2):
        """Perform sparse convolution over coefficient dictionaries."""
        result = defaultdict(complex)
        for k1, v1 in dict1.items():
            for k2, v2 in dict2.items():
                k_sum = tuple(np.add(k1, k2))
                result[k_sum] += v1 * v2
        return dict(result)

    def __mul__(self, other):
        """Multiply two Fourier series or scale by a scalar."""
        if isinstance(other, FourierSeriesND):
            dict1 = self.to_dict()
            dict2 = other.to_dict()
            result_dict = self._sparse_convolve(dict1, dict2)
            max_order = self.max_order or other.max_order
            max_k = min(self.max_k, other.max_k) if self.max_k and other.max_k else self.max_k or other.max_k
            numerical_tol = min(self.numerical_tol, other.numerical_tol) if self.numerical_tol is not None else None
            return FourierSeriesND(result_dict,dim=self.dim, max_order=max_order, max_k=max_k, numerical_tol=numerical_tol,mp=self.mp,complex_basis=self.complex_basis)
        elif isinstance(other, (int, float, complex,self.mp.Basic)):
            coeff_dict = {k: v * other for k, v in self.to_dict().items()}
            return FourierSeriesND(coeff_dict,dim=self.dim, max_order=self.max_order, max_k=self.max_k, numerical_tol=self.numerical_tol,mp=self.mp,complex_basis=self.complex_basis)
        else:
            return NotImplemented
        
    def __truediv__(self, other):
        if isinstance(other, (int, float, complex)):
            return self.__mul__(1 / other)
        else:
            return NotImplemented

    def __rmul__(self, other):
        """Support scalar * FourierSeriesND multiplication."""
        return self.__mul__(other)
    
    def __radd__(self, other):
        """Support scalar + FourierSeriesND addition."""
        return self.__add__(other)
    
    def __rsub__(self, other):
        """Support scalar - FourierSeriesND."""
        # other is expected to be a scalar (int/float/complex/numpy scalar/sympy number)
        # Compute: other + (-self)
        return (-self).__add__(other)

    def __pow__(self, power):
        """Raise the series to an integer power by repeated multiplication."""
        assert isinstance(power, int) and power >= 0
        identity = self.copy({(0,) * self.dim: 1.0})
        result = identity
        for _ in range(power):
            result = result * self
        return result
    
    def __sub__(self, other):
        """Subtract two FourierSeriesND instances."""
        return self + (-1 * other)
    
    def __neg__(self):
        """Unary negation: return -self."""
        coeff_dict = {k: -v for k, v in self.to_dict().items()}
        return self.copy(coeff_dict)

    def __getitem__(self, key):
        """Access a specific Fourier coefficient."""
        return self._coeff_dict.get(key, 0)
    
    def __repr__(self):
        """Compact string representation of active coefficients sorted by descending amplitude."""
        terms = self.to_dict()
        if self.mp is MathlibSympy:
            print('FourierSeriesND')
            for k, v in terms.items():
                self.mp.print_eq(k, v)
            return ''
        else:
            cleaned_terms = [(tuple(int(i) for i in k), v) for k, v in terms.items()]
            try:
                sorted_terms = sorted(cleaned_terms, key=lambda kv: -abs(kv[1]))
            except:
                sorted_terms = cleaned_terms
            return "FourierSeriesND(" + ", ".join(f"A{n}={v}" for n, v in sorted_terms) + ")"
        





