import numpy as np
from scipy.signal import fftconvolve
from collections import defaultdict
import pandas as pd

class FourierSeriesND:
    def __init__(self, coeff_dict, shape=None):
        """
        coeff_dict: dict mapping index tuples (n1, n2, ..., nd) to complex values
        shape: optional shape of the internal array (used to pad to full support)
        """
        self.dim = len(next(iter(coeff_dict))) if coeff_dict else 1
        self.offsets = np.array([0] * self.dim)

        # Find support range in each dimension
        if coeff_dict:
            mins = np.min([k for k in coeff_dict], axis=0)
            maxs = np.max([k for k in coeff_dict], axis=0)
            self.offsets = -mins
            full_shape = tuple((maxs - mins) + 1) if shape is None else shape
            self.data = np.zeros(full_shape, dtype=complex)

            for idx, val in coeff_dict.items():
                shifted = tuple(np.array(idx) + self.offsets)
                self.data[shifted] = val
        else:
            self.data = np.zeros((1,) * self.dim, dtype=complex)

    def to_dict(self, tol=1e-12):
        """Return dict {(n1, ..., nd): A_n} with non-negligible values"""
        it = np.nditer(self.data, flags=['multi_index'])
        result = {}
        for val in it:
            if abs(val) > tol:
                idx = tuple(np.array(it.multi_index) - self.offsets)
                result[idx] = val.item()
        return result

    def coeffs(self, tol=1e-12):
        """Return active coefficients as a pandas DataFrame for easy inspection"""
        coeff_dict = self.to_dict(tol)
        data = [(k, v) for k, v in coeff_dict.items()]
        df = pd.DataFrame(data, columns=["mode", "coefficient"])
        return df.sort_values(by="mode")

    def truncate(self, max_modes=None, top_k=None):
        """
        Truncate the Fourier series.

        Parameters:
            max_modes: tuple (n1_max, n2_max, ...) specifying max |n_i| in each dim
            top_k: keep only top_k largest coefficients by absolute value
        Returns:
            A new truncated FourierSeriesND
        """
        coeff_dict = self.to_dict()
        if max_modes:
            coeff_dict = {
                k: v for k, v in coeff_dict.items()
                if all(abs(k[i]) <= max_modes[i] for i in range(len(max_modes)))
            }
        if top_k is not None:
            coeff_items = sorted(coeff_dict.items(), key=lambda kv: -abs(kv[1]))
            coeff_dict = dict(coeff_items[:top_k])
        return FourierSeriesND(coeff_dict)

    def star(self):
        """Return the complex conjugate Psi^* with coefficients A_{-n}^*"""
        result = FourierSeriesND({}, shape=self.data.shape)
        result.data = np.conj(np.flip(self.data))
        result.offsets = -self.offsets.copy()
        return result

    def conj(self):
        """Alias for star()"""
        return self.star()

    def _aligned_with(self, other):
        """Return aligned data arrays and common offset for addition"""
        new_offsets = np.maximum(self.offsets, other.offsets)
        new_shape = np.maximum(
            np.array(self.data.shape) + (new_offsets - self.offsets),
            np.array(other.data.shape) + (new_offsets - other.offsets)
        )

        A1 = np.zeros(new_shape, dtype=complex)
        A2 = np.zeros(new_shape, dtype=complex)

        slices1 = tuple(slice(o, o + s) for o, s in zip(new_offsets - self.offsets, self.data.shape))
        slices2 = tuple(slice(o, o + s) for o, s in zip(new_offsets - other.offsets, other.data.shape))
        A1[slices1] = self.data
        A2[slices2] = other.data

        return A1, A2, new_offsets

    def __add__(self, other):
        A1, A2, new_offsets = self._aligned_with(other)
        result = FourierSeriesND({}, shape=A1.shape)
        result.data = A1 + A2
        result.offsets = new_offsets
        return result

    def __mul__(self, other):
        if isinstance(other, FourierSeriesND):
            result_data = fftconvolve(self.data, other.data, mode='full')
            result_offsets = self.offsets + other.offsets

            result = FourierSeriesND({}, shape=result_data.shape)
            result.data = result_data
            result.offsets = result_offsets
            return result
        elif isinstance(other, (int, float, complex)):
            result = FourierSeriesND({}, shape=self.data.shape)
            result.data = self.data * other
            result.offsets = self.offsets.copy()
            return result
        else:
            return NotImplemented

    def __rmul__(self, other):
        # scalar * self
        return self.__mul__(other)

    def __pow__(self, power):
        assert isinstance(power, int) and power >= 0
        identity = FourierSeriesND({(0,) * self.dim: 1.0})
        result = identity
        for _ in range(power):
            result = result * self
        return result

    def __repr__(self):
        terms = self.to_dict()
        return "FourierSeriesND(" + ", ".join(f"A{n}={v:.3g}" for n, v in sorted(terms.items())) + ")"
