import numpy as np
from collections import defaultdict
import pandas as pd

class FourierSeriesND:
    def __init__(self, coeff_dict, max_modes=None, max_k=500, content_fraction=None, numerical_tol=1e-21,bet0=1):
        """
        Initialize a FourierSeriesND object using sparse representation.

        Parameters:
            coeff_dict (dict): Mapping from index tuples (n1, n2, ..., nd) to complex values.
            max_modes (tuple, optional): Truncate to modes where |n_i| <= max_modes[i].
            max_k (int): Keep only the max_k largest coefficients by magnitude.
            content_fraction (float): Fraction of total energy to retain.            
            numerical_tol (float): Threshold for filtering near-zero values.
            bet0 (float): rescaling length for the Fourier series (psi has units of sqrt(m)!).
        """
        self.dim = len(next(iter(coeff_dict))) if coeff_dict else 1
        self.max_modes = max_modes
        self.max_k = max_k
        self.content_fraction = content_fraction
        self.numerical_tol = numerical_tol
        self.bet0 = bet0

        if max_modes is not None or max_k is not None:
            coeff_dict = self._truncate_dict(coeff_dict, max_modes, max_k,content_fraction)

        self._coeff_dict = {k: v for k, v in coeff_dict.items() if v != 0}


    def copy(self):
        """Return a new copy of the Fourier series."""
        coeff_dict = {k: v for k, v in self._coeff_dict.items()}
        return FourierSeriesND(coeff_dict, max_modes=self.max_modes, max_k=self.max_k, numerical_tol=self.numerical_tol)

    def _truncate_dict(self, coeff_dict, max_modes=None, max_k=None, content_fraction=None):
        """
        Truncate the Fourier series.

        Parameters:
            max_modes (tuple): Max absolute value of each mode component.
            max_k (int): Keep only the max_k largest coefficients by magnitude.
        """
        if max_modes:
            if isinstance(max_modes, int):
                max_modes = (max_modes,) * self.dim
                
            coeff_dict = {
                k: v for k, v in coeff_dict.items()
                if all(abs(k[i]) <= max_modes[i] for i in range(len(max_modes)))
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
        return df.sort_values(by="coefficient", key=lambda col: abs(col), ascending=False).reset_index(drop=True)


    def truncate(self, max_modes=None, max_k=None,content_fraction=None):
        """
        Returns a new FourierSeriesND with truncated coefficients.
        """
        return FourierSeriesND(self._coeff_dict, max_modes=max_modes, max_k=max_k,content_fraction=content_fraction, numerical_tol=self.numerical_tol)

    def star(self):
        """Return the complex conjugate series with reversed mode indices."""
        coeff_dict = self.to_dict()
        conjugated = {tuple(-np.array(k)): np.conj(v) for k, v in coeff_dict.items()}
        return FourierSeriesND(conjugated, max_modes=self.max_modes, max_k=self.max_k, numerical_tol=self.numerical_tol)

    def conj(self):
        """Alias for .star(), returns complex conjugate."""
        return self.star()

    
    def __add__(self, other):
        """Add two FourierSeriesND instances."""
        if isinstance(other, FourierSeriesND):
            dict1 = self.to_dict()
            dict2 = other.to_dict()
            all_keys = set(dict1.keys()) | set(dict2.keys())
            result_dict = {k: dict1.get(k, 0) + dict2.get(k, 0) for k in all_keys}
            max_modes = self.max_modes or other.max_modes
            max_k = min(self.max_k, other.max_k) if self.max_k and other.max_k else self.max_k or other.max_k
            numerical_tol = min(self.numerical_tol, other.numerical_tol)
            return FourierSeriesND(result_dict, max_modes=max_modes, max_k=max_k, numerical_tol=numerical_tol)
        elif isinstance(other, (int, float, complex)):
            dict1 = self.to_dict()
            result_dict = dict(dict1)
            zero_mode = (0,) * self.dim
            result_dict[zero_mode] = result_dict.get(zero_mode, 0) + complex(other)
            return FourierSeriesND(result_dict, max_modes=self.max_modes, max_k=self.max_k, numerical_tol=self.numerical_tol)
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
            max_modes = self.max_modes or other.max_modes
            max_k = min(self.max_k, other.max_k) if self.max_k and other.max_k else self.max_k or other.max_k
            numerical_tol = min(self.numerical_tol, other.numerical_tol)
            return FourierSeriesND(result_dict, max_modes=max_modes, max_k=max_k, numerical_tol=numerical_tol)
        elif isinstance(other, (int, float, complex)):
            coeff_dict = {k: v * other for k, v in self.to_dict().items()}
            return FourierSeriesND(coeff_dict, max_modes=self.max_modes, max_k=self.max_k, numerical_tol=self.numerical_tol)
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

    def __pow__(self, power):
        """Raise the series to an integer power by repeated multiplication."""
        assert isinstance(power, int) and power >= 0
        identity = FourierSeriesND({(0,) * self.dim: 1.0}, max_modes=self.max_modes, max_k=self.max_k, numerical_tol=self.numerical_tol)
        result = identity
        for _ in range(power):
            result = result * self
        return result
    
    def __sub__(self, other):
        """Subtract two FourierSeriesND instances."""
        return self + (-1 * other)

    
    
    def __repr__(self):
        """Compact string representation of active coefficients sorted by descending amplitude."""
        terms = self.to_dict()
        cleaned_terms = [(tuple(int(i) for i in k), v) for k, v in terms.items()]
        sorted_terms = sorted(cleaned_terms, key=lambda kv: -abs(kv[1]))
        return "FourierSeriesND(" + ", ".join(f"A{n}={v:.3g}" for n, v in sorted_terms) + ")"
    





