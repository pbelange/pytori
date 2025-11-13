import operator
from collections import defaultdict
import numpy as np
from functools import reduce

import pytori.mathlib as mathlib
import pytori.normalform as normalform


# Sympy-only method decorator
def sympy_only(func):
    def wrapper(self, *args, **kwargs):
        if getattr(self.mp, "name", None) != "sympy":
            raise TypeError(f"Method '{func.__name__}' requires mp='sympy' backend.")
        return func(self, *args, **kwargs)
    return wrapper

# Numpy-only method decorator
def numpy_only(func):
    def wrapper(self, *args, **kwargs):
        if getattr(self.mp, "name", None) != "numpy":
            raise TypeError(f"Method '{func.__name__}' requires mp='numpy' backend.")
        return func(self, *args, **kwargs)
    return wrapper

#=========================================================================================
# BASE series
#=========================================================================================
class BaseSeries:
    def __init__(self, coeff_dict = {}, dim = None, max_order=None, max_terms = 100, numerical_tol = None, mp='numpy'):

        # Infer dimension
        #-------------------------
        if len(coeff_dict) == 0:
            assert dim is not None, "Cannot create empty series without specifying dimension."
            self.dim = dim
        else:
            nterm = next(iter(coeff_dict))
            assert isinstance(nterm, tuple), "Coefficient keys must be tuples representing mode indices."
            inferred_dim = len(nterm)
            if dim is not None:
                assert dim == inferred_dim, \
                    f"Specified dim ({dim}) does not match inferred dimension ({inferred_dim})"
            self.dim = inferred_dim
        #-------------------------

        # Extract parameters
        self.max_order      = max_order
        self.max_terms      = max_terms
        self.numerical_tol  = numerical_tol
        self.mp             = mp

        # Backend-specific adjustments
        if self.mp.name == 'sympy':
            self.max_terms      = None  # disable max_terms for sympy to avoid ordering issues
            self.numerical_tol  = None  # disable numerical_tol for sympy to avoid symbolic comparisons

        
        # Clean up
        if max_order is not None or max_terms is not None or numerical_tol is not None:
            coeff_dict = self._truncate_dict(coeff_dict, max_order, max_terms, numerical_tol)

        # Store non-zero coefficients
        self._coeffs = {k: v for k, v in coeff_dict.items() if v != 0}

    
    #--------------------------------------------------------------------------------
    # Backend management
    @property
    def mp(self):
        return self._mp
    
    @mp.setter
    def mp(self, value):
        self._mp = mathlib.import_mathlib(value) if isinstance(value, str) else value
    #--------------------------------------------------------------------------------

    @property
    def coeffs(self):
        """Return internal coefficient dictionary directly."""
        return self._coeffs
    
    def _metadata(self):
        """
        Return minimal identifying metadata for this series.
        """
        mp_name = getattr(self.mp, "name", str(self.mp))
        return (self.dim, self.max_order, self.max_terms, self.numerical_tol, mp_name)

    def _truncate_dict(self, coeff_dict, max_order=None, max_terms=None, numerical_tol=None):
        """
        Filter and truncate a coefficient dictionary according to:
          1. numerical_tol    → drop coefficients |v| ≤ tol
          2. max_order        → restrict to allowed mode indices
          3. max_terms    → keep top-N largest coefficients by magnitude

        Parameters
        ----------
        coeff_dict : dict
            Mapping from index tuples to coefficient values.
        max_order : int | tuple | None
            If int, keeps modes where sum(|n_i|) <= max_order.
            If tuple, enforces |n_i| <= max_order[i] for each dimension.
        max_terms : int | None
            Maximum number of coefficients to retain (sorted by |value|).
        numerical_tol : float | None
            Drop coefficients with magnitude <= tol. If None, skip this filter.

        Returns
        -------
        dict
            Truncated coefficient dictionary.
        """
        mp = self.mp  # backend alias
        coeff_dict = dict(coeff_dict)  # avoid mutating input

        # ----------------------------------------------------------
        # 1. Drop coefficients below numerical tolerance
        # ----------------------------------------------------------
        if numerical_tol is not None:
            coeff_dict = {
                k: v for k, v in coeff_dict.items()
                if mp.abs(v) > numerical_tol
            }

        # ----------------------------------------------------------
        # 2. Truncate by mode order
        # ----------------------------------------------------------
        if max_order is not None:
            if isinstance(max_order, int):
                coeff_dict = {
                    k: v for k, v in coeff_dict.items()
                    if sum(abs(ki) for ki in k) <= max_order
                }
            elif isinstance(max_order, (tuple, list)):
                assert len(max_order) == self.dim, \
                    f"max_order (length {len(max_order)}) must match dimension ({self.dim})"
                coeff_dict = {
                    k: v for k, v in coeff_dict.items()
                    if all(abs(k[i]) <= max_order[i] for i in range(self.dim))
                }

        # ----------------------------------------------------------
        # 3. Keep only top-N harmonics
        # ----------------------------------------------------------
        if max_terms is not None and len(coeff_dict) > max_terms:
            # sort descending by absolute magnitude
            sorted_items = sorted(
                coeff_dict.items(),
                key=lambda kv: float(abs(kv[1])),
                reverse=True
            )
            coeff_dict = dict(sorted_items[:max_terms])

        return coeff_dict



    def to_dict(self):
        """
        Return a dictionary representation of the series, including metadata
        and coefficients.

        Returns
        -------
        dict
            Structured representation of the BaseSeries object.
        """
        d = {
            "dim": self.dim,
            "max_order": self.max_order,
            "max_terms": self.max_terms,
            "numerical_tol": self.numerical_tol,
            "backend": getattr(self.mp, "name"),
            "coeffs":dict(self._coeffs)
        }
        return d


    def copy(self, coeff_dict=None, **overrides):
        """
        Return a deep copy of this series, optionally with new coefficients
        and/or updated configuration parameters.
        """
        cls = self.__class__

        # Reuse existing internal dict safely (it's already independent)
        # Copy is done at __init__ already
        coeffs = coeff_dict if coeff_dict is not None else self._coeffs

        params = dict(
            coeff_dict=coeffs,
            dim=self.dim,
            max_order=self.max_order,
            max_terms=self.max_terms,
            numerical_tol=self.numerical_tol,
            mp=self.mp,
        )

        if overrides:
            params.update(overrides)

        return cls(**params)



    def truncate(self, max_order=None, max_terms=None, numerical_tol=None):
        """
        Return a truncated copy of this series.

        Filter and truncate a coefficient dictionary according to:
          1. numerical_tol    → drop coefficients |v| ≤ tol
          2. max_order        → restrict to allowed mode indices
          3. max_terms    → keep top-N largest coefficients by magnitude
        """
        # Explicit defaults
        max_order      = self.max_order      if max_order      is None else max_order
        max_terms  = self.max_terms  if max_terms  is None else max_terms
        numerical_tol  = self.numerical_tol  if numerical_tol  is None else numerical_tol
        return self.copy(
            max_order=max_order,
            max_terms=max_terms,
            numerical_tol=numerical_tol,
        )
    

    def _merge_params(self, other=None, **overrides):
        """
        Merge parameters conservatively, returning None if metadata is identical.

        Parameters
        ----------
        other : BaseSeries | None
            Optional other series to merge parameters with.
        **overrides :
            Optional keyword overrides.

        Returns
        -------
        dict | None
            - None  → if no merge is needed (metadata identical).
            - dict  → merged parameters for constructor.
        """
        # --- Fast path -----------------------------------------------------
        if other is None or self._metadata() == other._metadata():
            if not overrides:
                return None
            return dict(
                dim=self.dim,
                max_order=self.max_order,
                max_terms=self.max_terms,
                numerical_tol=self.numerical_tol,
                mp=self.mp,
                **overrides,
            )

        # --- Slow path -----------------------------------------------------
        if self.dim != other.dim:
            raise ValueError("Cannot merge series with different dimensions.")
        if self.mp != other.mp:
            raise ValueError("Cannot merge series with different backends.")

        def _min_or_none(a, b):
            if a is None: return b
            if b is None: return a
            if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
                return tuple(min(ai, bi) for ai, bi in zip(a, b))
            return min(a, b)

        merged = dict(
            dim=self.dim,
            max_order=_min_or_none(self.max_order, other.max_order),
            max_terms=_min_or_none(self.max_terms, other.max_terms),
            numerical_tol=_min_or_none(self.numerical_tol, other.numerical_tol),
            mp=self.mp,
            **overrides,
        )
        return merged

    #=====================================================================================
    # Arithmetic operations
    #=====================================================================================
    def __neg__(self):
        """Unary negation: returns -self."""
        coeffs = {k: -v for k, v in self.coeffs.items()}
        return self.copy(coeff_dict=coeffs)
    
    def __pos__(self):
        """Unary plus: returns self unchanged."""
        return self

    def __add__(self, other):
        """
        Add two BaseSeries (or subclass) instances, or add a scalar offset.

        Returns
        -------
        BaseSeries (or subclass)
            A new instance containing the summed coefficients.
        """

        # --- Case 1: same class ---
        if type(self) is type(other):
            merged = self._merge_params(other)
            all_keys = set(self.coeffs) | set(other.coeffs)
            coeffs = {k: self.coeffs.get(k, 0) + other.coeffs.get(k, 0) for k in all_keys}
            return self.copy(coeff_dict=coeffs, **(merged or {}))

        # --- Case 2: scalar addition ---
        elif isinstance(other, mathlib.SCALARS):
            coeffs = dict(self.coeffs)
            zero_mode = (0,) * self.dim
            coeffs[zero_mode] = coeffs.get(zero_mode, 0) + other
            return self.copy(coeff_dict=coeffs)

        # --- otherwise, not supported ---
        return NotImplemented


    def _sparse_convolve(self, dict1, dict2):
        """
        Sparse convolution of two coefficient dictionaries.

        For each (k1, v1) in dict1 and (k2, v2) in dict2:
            result[k1 + k2] += v1 * v2

        Returns
        -------
        dict
            Convolved coefficient dictionary.
        """
        mp = self.mp  # backend alias
        result = defaultdict(lambda: 0 if mp.name == "numpy" else mp.sympify(0))
        for k1, v1 in dict1.items():
            for k2, v2 in dict2.items():
                k_sum = tuple(map(operator.add, k1, k2))
                result[k_sum] += v1 * v2
        return dict(result)


    def __mul__(self, other):
        """
        Multiply two BaseSeries (or subclass) instances, or scale by a scalar.

        Returns
        -------
        BaseSeries (or subclass)
            A new instance containing the product coefficients.
        """

        # --- Case 1: same class ---
        if type(self) is type(other):
            merged = self._merge_params(other)
            coeffs = self._sparse_convolve(self.coeffs, other.coeffs)
            return self.copy(coeff_dict=coeffs, **(merged or {}))

        # --- Case 2: scalars   ---
        elif isinstance(other, mathlib.SCALARS):
            coeffs = {k: v * other for k, v in self.coeffs.items()}
            return self.copy(coeff_dict=coeffs)

        # --- otherwise, not supported ---
        return NotImplemented


    def conjugate(self):
        """Returns element-wise conjugation."""
        mp = self.mp  # backend alias
        coeffs = {k: mp.conjugate(v) for k, v in self.coeffs.items()}
        return self.copy(coeff_dict=coeffs)
    
    def conj(self):
        """Alias for .conjugate(), returns complex conjugate."""
        return self.conjugate()
    
    


    # Variations of __add__ 
    #----------------------------------------------
    def __radd__(self, other):
        """Support scalar + series addition."""
        return self.__add__(other)
    def __sub__(self, other):
        """Subtract two series instances."""
        return self + (-1 * other)
    def __rsub__(self, other):
        """Support scalar - series."""
        return (-self).__add__(other)
    #----------------------------------------------

    # Variations of __mul__ 
    #----------------------------------------------
    def __rmul__(self, other):
        """Support scalar * series multiplication."""
        return self.__mul__(other)
    
    def __truediv__(self, other):
        """Support series/scalar divisions."""
        if isinstance(other, mathlib.SCALARS):
            return self.__mul__(1 / other)
        return NotImplemented
    
    def __pow__(self, power):
        """Raise the series to an integer power by repeated multiplication."""
        assert isinstance(power, int) and power >= 0, "Power must be a positive integer."

        identity = self.copy(coeff_dict={(0,) * self.dim: 1.0})

        if power == 0:
            return identity

        result = identity
        for _ in range(power):
            result = result * self
        return result
    #----------------------------------------------

    #=====================================================================================
    # Symbolic methods
    #=====================================================================================
    @sympy_only
    def subs(self, subs):
        """Substitute symbolic variables in the coefficients."""
        substituted = {k: v.subs(subs) if hasattr(v, 'subs') else v for k, v in self.coeffs.items()}
        return self.copy(coeff_dict=substituted)
    
    @sympy_only
    def evalf(self):
        """Numerical evaluation of symbolic coefficients."""
        evaluated = {k: complex(v.evalf()) if hasattr(v, 'evalf') else v for k, v in self.coeffs.items()}
        return self.copy(coeff_dict=evaluated)
    
    @sympy_only
    def simplify(self):
        """Simplify symbolic coefficients (lightweight)."""
        mp = self.mp
        simplified = {
            k: mp.simplify(v) if hasattr(v, "simplify") else v
            for k, v in self.coeffs.items()
        }
        return self.copy(coeff_dict=simplified)
    

    @sympy_only
    def nsimplify(self):
        """Conversion of floats to symbolic coefficients (lightweight)."""
        mp = self.mp
        simplified = {
            k: mp.simplify(mp.nsimplify(v)) for k, v in self.coeffs.items()
        }
        return self.copy(coeff_dict=simplified)
    
    @sympy_only
    def explicit(self, base_symbols):
        """
        Return explicit symbolic expansion of the multivariate series:
            Ψ = ∑_k c_k ∏_j base_symbols[j]**k[j]
        """
        mp = self.mp
        expr = 0
        for k, v in self.coeffs.items():
            term = reduce(operator.mul, (base**nk for base, nk in zip(base_symbols, k)), 1)
            expr += v * term

        return mp.expand(expr)
    

    @sympy_only
    def symbols_set(self):
        """Return the set of all symbolic variables used in the coefficients."""
        return set().union(*(term.free_symbols for term in self.coeffs.values()))

    #=====================================================================================
    # Printing methods
    #=====================================================================================
    def __getitem__(self, key):
        """Access a specific Fourier coefficient."""
        return self._coeffs.get(key, 0)
    

    def __repr__(self):
        try:
            from pytori.display import render_series
            return render_series(self)
        except Exception:
            from pytori.display import render_series_ascii
            return render_series_ascii(self)

#=========================================================================================
# FOURIER SERIES
#=========================================================================================
class FourierSeries(BaseSeries):
    """
    Multidimensional Fourier series representation:
        Ψ(θ) = Σₖ Aₖ · exp(i k·θ)

    Inherits all algebraic and truncation functionality from BaseSeries,
    and adds domain-specific methods for FourierSeries

    Attributes
    ----------
    coeffs : dict[tuple[int], complex]
        Mapping from integer mode tuples k to complex coefficients Aₖ.
    dim : int
        Dimension of the angular space (len(k)).
    mp : module
        Math backend (numpy, sympy, or mpmath) supporting exp, conj, etc.
    """

    def __init__(self, coeff_dict={}, **kwargs):
        super().__init__(coeff_dict=coeff_dict, **kwargs)


    # -------------------------------------------------------------------------
    def conjugate(self):
        """
        Complex conjugate of the Fourier series.

        For Fourier modes exp(i k·θ), conjugation implies:
            Aₖ → A₋ₖ*,   exp(i k·θ) → exp(-i k·θ)

        Returns
        -------
        FourierSeries
            The conjugated Fourier series.
        """
        mp = self.mp
        conj_coeffs = {
            tuple(map(operator.neg, k)): mp.conjugate(v)
            for k, v in self.coeffs.items()
        }
        return self.copy(coeff_dict=conj_coeffs)
    
    def dephase(self, phases):
        """
        Dephase the Fourier series by shifting the phases of each dimension.

        Parameters
        ----------
        phases : float | sequence of float
            Phase shifts for each dimension in radians.
        """
        mp = self.mp

        if isinstance(phases, mathlib.SCALARS):
            phases = [phases] * self.dim
        assert len(phases) == self.dim, "Length of phases must match dimension"

        phased_dict = {
            k: v * mp.exp(mp.I * sum(ki * phi for ki, phi in zip(k, phases)))
            for k, v in self.coeffs.items()
        }
        return self.copy(coeff_dict=phased_dict)
    

    @numpy_only
    def eval(self, angles):
        """
        Evaluate the Fourier series Ψ(Θ) = Σₙ Aₙ e^{i (n·Θ)} at given angles (NumPy only).

        Parameters
        ----------
        angles : float | sequence of float | sequence of np.ndarray
            Angles Θ for each dimension.
            Missing dimensions are padded with zeros.

            Example:
                angles = [theta_x, theta_y] or [theta_x]  (Θ_y=0 assumed)

        Returns
        -------
        ndarray or complex
            Evaluated Ψ(Θ). 
        """

        mp  = self.mp  # should be numpy backend
        dim = self.dim

            
        # Normalize angles
        if isinstance(angles, np.ndarray):
            if dim == 1:
                angles = [angles]
            else:
                raise ValueError("angles must be a sequence for dim > 1")
            
        elif isinstance(angles, (list, tuple)):
            # Trim or pad to match dimensionality
            if len(angles) < dim:
                angles = list(angles) + [0] * (dim - len(angles))
            elif len(angles) > dim:
                angles = list(angles)[:dim]  # safely ignore extras
        else:
            raise TypeError("angles must be scalar, np.ndarray, or sequence thereof")

        # --- Shape consistency check ---
        shapes = [np.shape(a) for a in angles if np.ndim(a) > 0]
        if len(set(shapes)) > 1:
            raise ValueError(f"Inconsistent angle array shapes: {shapes}")

        # Target shape for result
        target_shape = shapes[0] if shapes else ()
        result = np.zeros(target_shape, dtype=complex)

        # --- Evaluate Ψ(Θ) ---
        for n_vec, A in self.coeffs.items():
            phase = sum(ni * np.array(theta) for ni, theta in zip(n_vec, angles))
            result += A * np.exp(1j * phase)

        return result
            



#=========================================================================================
# Normal Form Series
#=========================================================================================
class NormalFormSeries(BaseSeries):
    """
    Multivariate complex power series in (p, p*) pairs:
        Ψ(p, p*) = Σ_{j,k,l,m,p,q,...} a_{jklmpq} 
                    (p_x^j p_x^{*k})(p_y^l p_y^{*m})(p_ζ^p p_ζ^{*q}) ...

    Inherits full algebraic and truncation functionality from BaseSeries
    but redefines conjugation to exchange each (, p*) index pair.

    This corresponds to:
        a_{j,k,l,m,p,q}  →  a_{k,j,m,l,q,p}*
    """

    def __init__(self, coeff_dict={}, **kwargs):
        super().__init__(coeff_dict=coeff_dict, **kwargs)
        assert self.max_order is not None, "NormalFormSeries requires max_order to be set."
        assert self.dim % 2 == 0, "NormalFormSeries dimension must be even ((p, p*) pairs)"

    # -------------------------------------------------------------------------
    def conjugate(self):
        """
        Complex conjugate of the Normal Form series.

        For each (p, p*) pair, indices are swapped:
            (j,k,l,m,p,q,...) → (k,j,m,l,q,p,...)
        and coefficients are complex conjugated.
        """
        mp = self.mp
        conjugated = {}

        for k, v in self.coeffs.items():
            # Swap each consecutive pair: (0,1), (2,3), ...
            swapped_k = tuple(k[i+1] if i % 2 == 0 else k[i-1] for i in range(len(k)))
            conjugated[swapped_k] = mp.conjugate(v)

        return self.copy(coeff_dict=conjugated)



    # -------------------------------------------------------------------------
    def dephase(self, omega_dicts):
        """
        Apply amplitude-dependent dephasing to this NormalFormSeries.

        p_j → exp(i Q_j(I)) p_j

        Parameters
        ----------
        omega_dicts : dict[str, dict]
            Per-plane detuning expansions. For example:
            {
                'x': {(0,): ω₀x, (1,): ω₁x, (2,): ω₂x, ...},
                'y': {(0,): ω₀y, (1,): ω₁y, ...},
            }

        Returns
        -------
        NormalFormSeries
            Dephased series Ψ(ρ) → Ψ(e^{iQ(ρρ*)}ρ)
        """

        dim = self.dim
        P   = dim // 2
        mp  = self.mp

        # Build the rho_j transforms
        f_list = []
        plane_key = {'x':0, 'y':1, 'z':2}
        for key_j,omega_j in omega_dicts.items():
            j   = plane_key[key_j]
            fj  = normalform.rho_transform(j, omega_j, order=self.max_order, mp=mp)
            f_list.append(fj)

        out = self.copy(coeff_dict={})  # initialize empty

        # Apply substitution term-by-term
        for k_tuple, coeff in self.coeffs.items():
            term = coeff
            for j in range(P):
                rho_j_term = (f_list[j] ** k_tuple[2*j]) * (f_list[j].conj() ** k_tuple[2*j+1])
                term = term * rho_j_term
            out += term

        if mp.name == "sympy":
            out = out.nsimplify()
        return out
    

    def symplectic_condition(self, subs=None, as_equations=False):
        """
        Compute the multi-plane (6D-ready) symplectic condition for a NormalFormSeries.

        The symplecticity requirement for Ψ(p, p*) reads:
            J = Σ_j [ (∂Ψ/∂p_j)(∂Ψ*/∂p_j*) - (∂Ψ/∂p_j*)(∂Ψ*/∂p_j) ]  =  1

        This condition is evaluated algebraically in coefficient space using
        the NormalFormSeries operations (differentiation, conjugation, multiplication).

        Parameters
        ----------
        subs : dict, optional
            Optional SymPy substitutions applied to the final coefficients
            before simplification (e.g., {omega_0: mu}).
        as_equations : bool, default=False
            - If False (default): returns the NormalFormSeries J - 1, representing
            the symplectic residual.
            - If True: returns a list of SymPy equations Eq(coeff, 0) enforcing
            symplecticity term by term.

        Returns
        -------
        NormalFormSeries or list[sympy.Eq]
            If as_equations=False (default):
                The truncated NormalFormSeries representing J - 1.
            If as_equations=True:
                A list of SymPy equations, each requiring a coefficient of (J - 1)
                to vanish identically.

        Notes
        -----
        - The implementation is fully algebraic and backend-agnostic.
        - Requires mp='sympy' backend when `as_equations=True`.
        - Works for any even-dimensional NormalFormSeries (2D, 4D, 6D, …).
        - Uses self.max_order as truncation limit to control combinatorial growth.
        """
        mp  = self.mp
        dim = self.dim

        work_order = self.max_order
        assert work_order is not None and work_order >= 0

        # Initialize J = 0
        J = self.copy(coeff_dict={(0,) * dim: 0})

        # Build symplectic sum over planes
        for j in range(dim // 2):
            dP_dr     = normalform._partial_of(self, j, by="rho")
            dP_drs    = normalform._partial_of(self, j, by="rho*")
            dPs_dr    = normalform._partial_of(self.conj(), j, by="rho")
            dPs_drs   = normalform._partial_of(self.conj(), j, by="rho*")

            J += (dP_dr * dPs_drs) - (dPs_dr * dP_drs)

        # Enforce J == 1 -> J-1 ==0
        J_minus_1 = (J - 1).truncate(max_order=work_order-1)
        
        # Optionally return symbolic equations
        if as_equations:
            if mp.name != "sympy":
                raise TypeError("as_equations=True requires mp='sympy' backend.")
            import sympy as sp
            equations = []
            for k, c in J_minus_1.coeffs.items():
                cc = c.subs(subs) if subs else c
                cc = mp.simplify(cc)
                equations.append(sp.Eq(cc, 0))
            return equations

        # Default: return the algebraic J − 1 series
        return J_minus_1
    
    def symplectic_residual(self):
        """Return numeric norm of symplectic deviation |J - 1|."""
        Jm1 = self.symplectic_condition(as_equations=False)
        mp  = self.mp
        return sum([mp.abs(v) for v in Jm1.coeffs.values()])

    def collapse(self, I, max_order=None, max_terms=100, numerical_tol=None, mp='numpy'):
        """
        Collapse a NormalFormSeries Ψ(ρ,ρ*) into a FourierSeries Ψ(Θ)
        by evaluating the radial dependence at fixed actions I.

            (ρ_j, ρ_j*) → (√(2I_j)e^{iΘ_j}, √(2I_j)e^{-iΘ_j})

        Parameters
        ----------
        I : float | complex | sequence
            Action(s) at which to evaluate the amplitude dependence.
            - scalar (float, complex, or sympy) valid only for 1D (dim=2)
            - iterable of length = dim/2 for multi-plane systems
        max_order : int | None, optional
            Truncation order for the resulting FourierSeries.
            Defaults to self.max_order if not specified.
        max_terms : int | None, optional
            Maximum number of terms to retain in the collapsed FourierSeries.
            Defaults to self.max_terms if not specified.
        numerical_tol : float | None, optional
            Minimum absolute magnitude threshold for coefficients to keep.
            Defaults to self.numerical_tol if not specified.

        Returns
        -------
        FourierSeries
            Collapsed Fourier representation:
                Ψ(Θ) = Σₙ Aₙ · e^{i (n·Θ)},
            where each Aₙ = Σ_jk... a_{jk...} (2I_x)^{(j+k)/2} (2I_y)^{(l+m)/2} ...

        Notes
        -----
        - The output dimension is dim/2 (number of (ρ,ρ*) pairs).
        - The resulting Fourier coefficients depend on the chosen I values.
        """
        # --- Default overrides ---
        new_mp          = mp
        max_order       = max_order
        max_terms       = max_terms
        numerical_tol   = numerical_tol
        

        mp  = self.mp
        dim = self.dim
        P   = dim // 2

        # --- Normalize I input ---
        if isinstance(I, mathlib.SCALARS):
            assert P == 1, "Scalar I only valid for 1D (dim=2)."
            I = [I]
        I = list(I)
        assert len(I) == P, f"Expected {P} action values, got {len(I)}."

        

        # --- Initialize Fourier series ---
        from pytori.series import FourierSeries  # local import to avoid circular deps
        collapsed = FourierSeries(
            coeff_dict={},
            dim=P,
            max_order=max_order,
            max_terms=max_terms,
            numerical_tol=numerical_tol,
            mp=new_mp,
        )

        # --- Collapse each NormalForm term ---
        for k_tuple, coeff in self.coeffs.items():
            n_vec = []
            amp_factor = coeff

            for j in range(P):
                m = k_tuple[2 * j]
                n = k_tuple[2 * j + 1]

                # harmonic index in Θ_j
                n_vec.append(m - n)

                # amplitude term (2 I_j)^{(m+n)/2}
                if (m + n) != 0:
                    amp_factor *= (2 * I[j]) ** ((m + n) / 2)

            n_vec = tuple(n_vec)

            # accumulate in Fourier coefficients
            if n_vec in collapsed.coeffs:
                collapsed.coeffs[n_vec] += amp_factor
            else:
                collapsed.coeffs[n_vec] = amp_factor

        # --- Apply truncation / cleanup ---
        collapsed = collapsed.truncate(
            max_order=max_order,
            max_terms=max_terms,
            numerical_tol=numerical_tol,
        )

        return collapsed
