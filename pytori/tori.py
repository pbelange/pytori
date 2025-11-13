import numpy as np
from collections import defaultdict
from pytori.series import BaseSeries, FourierSeries, NormalFormSeries
import pytori.mathlib as mathlib
from pytori.display import render_torus

#===========================================================
# Helper: dimension inference
#===========================================================
def _infer_dim(self):
    """Infer consistent dimension among provided series."""
    
    series_list = [self._x, self._y, self._z]
    dims = []
    for s in series_list:
        if s is None:
            continue
        if isinstance(s, NormalFormSeries):
            dims.append(s.dim // 2)
        else:
            dims.append(s.dim)
    if not dims:
        raise ValueError("Cannot infer dimension from empty Torus.")
    dim = dims[0]
    assert all(d == dim for d in dims), "All series must have compatible dimensions."
    return dim

def _infer_class(self):
    cls_type = next(
        (s.__class__ for s in (self._x, self._y, self._z) if s is not None),
        "UnknownSeries",
    )
    return cls_type

def _init_series(plane_input, name, max_order, max_terms, numerical_tol, mp, series_class):
    """Initialize one plane, from dict or Series object."""
    
    if plane_input is None:
        return None
    
    if isinstance(plane_input, BaseSeries):
        return plane_input
    
    if isinstance(plane_input, dict):
        return series_class(
            coeff_dict=plane_input,
            max_order=max_order,
            max_terms=max_terms,
            numerical_tol=numerical_tol,
            mp=mp,
        )
    raise TypeError(f"{name} must be a dict, Series object, or None, not {type(plane_input)}")


#===========================================================
# Torus container
#===========================================================

class Torus:
    """
    Container for up to three series objects representing x, y, z planes.

    Supports both FourierSeries (Ψ(Θ)) and NormalFormSeries (Ψ(ρ,ρ*)).
    Provides clear, backend-robust evaluation of invariants and integrals.
    """
    DEFAULT_MAX_TERMS = 100
    def __init__(self, x=None, y=None, z=None,
                 bet0=(1, 1, 1),
                 max_order=None,
                 max_terms=DEFAULT_MAX_TERMS,
                 numerical_tol=None,
                 mp="numpy",
                 series_class=FourierSeries):

        # --------------------------------------------------------
        # 1. Initialize planes (dict → series)
        # --------------------------------------------------------
        
        self._x = _init_series(x, "x", max_order, max_terms, numerical_tol, mp, series_class)
        self._y = _init_series(y, "y", max_order, max_terms, numerical_tol, mp, series_class)
        self._z = _init_series(z, "z", max_order, max_terms, numerical_tol, mp, series_class)
        self.series_class = _infer_class(self)

        # --------------------------------------------------------
        # 3. Verify all planes share same backend
        # --------------------------------------------------------
        self.mp = self._x.mp if self._x is not None else mathlib.import_mathlib(mp)
        for s, name in zip([self._x, self._y, self._z], ["x", "y", "z"]):
            if s is None:
                continue
            if s.mp.name != self.mp.name:
                raise ValueError(
                    f"Inconsistent backends: plane '{name}' uses {s.mp.name}, "
                    f"expected {self.mp.name}."
                )

        # --------------------------------------------------------
        # 4. Infer dimension from provided series
        # --------------------------------------------------------
        self._dim = _infer_dim(self)

        # --------------------------------------------------------
        # 5. Store β-like scaling factors (optional)
        # --------------------------------------------------------
        self.betx0, self.bety0, self.betz0 = bet0

        # --------------------------------------------------------
        # 6. Cached attributes
        # --------------------------------------------------------
        self._needs_refresh = True
        self._Ax = self._Ay = self._Az = []
        self._nx = self._ny = self._nz = []


    @classmethod
    def from_naff(cls,n=None,A=None, max_terms=None, **kwargs):
        """
        Alternative constructor from (n, A) pairs, e.g., from NAFF output.
        Automatically handles aliasing by summing contributions with identical base frequency vectors.
        """
        assert len(A) == len(n), "A and n must have the same length"
        

        def _accumulate_modes(n_list, A_list, dim):
            """
            Helper to sum coefficients with the same aliased index n[:dim].
            """
            acc = defaultdict(complex)
            # Sanitize input
            n_list = [tuple(int(_n) for _n in n) for n in n_list]  # cast to int tuples
            A_list = [complex(a) for a in A_list]                  # ensure complex type

            for n, A in zip(n_list, A_list):
                acc[tuple(n[:dim])] += A

            return dict(acc)


        # Initialize
        x = y = z = None  
        dim = len(A)

        if dim >= 1:
            x = _accumulate_modes(n[0], A[0], dim)
        if dim >= 2:
            y = _accumulate_modes(n[1], A[1], dim)
        if dim >= 3:
            z = _accumulate_modes(n[2], A[2], dim)

        # --------------------------------------------------------
        # Expand the max_terms **only if NAFF produces more terms**
        # --------------------------------------------------------
        if max_terms is None:
            # incoming dictionary sizes
            sizes = [
                len(n[0]) if x else 0,
                len(n[1]) if y else 0,
                len(n[2]) if z else 0,
            ]
            naff_max = max(sizes)

            # use Torus default = 100
            default_max = cls.DEFAULT_MAX_TERMS

            # final max_terms (only expand; never shrink)
            max_terms = max(default_max, naff_max)

        return cls(x=x, y=y, z=z, max_terms=max_terms, **kwargs)
        
    

    @classmethod
    def linear_match(cls,I,lambda_plus=None, lambda_minus=None, W_matrix=None,
                    nemitt_x=None, nemitt_y=None, nemitt_z=None,
                    particle_on_co=None, beta_rel=None, gamma_rel=None):
        """
        Alternative constructor for a linear-matched Torus.
        Constructs a Torus representing a matched ellipse in each plane,
        based on provided Twiss parameters and emittances.
        """
        from pytori.transforms import norm2phys

        # Creating from Normal Form with fixed action
        #-------------------------------------------------------
        def basis_tuple(dim, plane_index):
            t = [0] * (2*dim)
            t[2 * plane_index] = 1
            return tuple(t)

        dim = len(I)
        _x = {basis_tuple(dim, 0): 1} if dim >= 1 else None
        _y = {basis_tuple(dim, 1): 1} if dim >= 2 else None
        _z = {basis_tuple(dim, 2): 1} if dim >= 3 else None

        CS_torus = cls(x=_x,y=_y,z=_z,max_order=1,series_class=NormalFormSeries)
        CS_torus = CS_torus.collapse(I)
        #-------------------------------------------------------

        # Cropping optics matrices for proper dimension
        #-------------------------------------------------------
        lambda_plus     = lambda_plus[:dim,:dim] if lambda_plus is not None else None
        lambda_minus    = lambda_minus[:dim,:dim] if lambda_minus is not None else None
        W_matrix        = W_matrix[:2*dim, :2*dim] if W_matrix is not None else None
        #-------------------------------------------------------

        # Converting to physical space:
        return norm2phys(CS_torus,
              lambda_plus=lambda_plus, lambda_minus=lambda_minus, W_matrix=W_matrix,
              nemitt_x=nemitt_x, nemitt_y=nemitt_y, nemitt_z=nemitt_z,
              particle_on_co=particle_on_co, beta_rel=beta_rel, gamma_rel=gamma_rel)


                



    #--------------------------------------------------------------------------------
    # Backend management
    @property
    def mp(self):
        return self._mp

    @mp.setter
    def mp(self, value):
        """Set and propagate the math backend to all planes."""
        self._mp = mathlib.import_mathlib(value) if isinstance(value, str) else value
        for s in (self._x, self._y, self._z):
            if s is not None:
                s.mp = self._mp

    # Sympy-only method decorator
    def sympy_only(func):
        """Decorator: restrict method to SymPy backend."""
        def wrapper(self, *args, **kwargs):
            if getattr(self.mp, "name", None) != "sympy":
                raise TypeError(f"Method '{func.__name__}' requires mp='sympy' backend.")
            return func(self, *args, **kwargs)
        return wrapper
    #--------------------------------------------------------------------------------


    @property
    def dim(self):
        return self._dim

    @property
    def needs_refresh(self):
        return self._needs_refresh
    
    #===========================================================
    # Cached property access
    #===========================================================
    # The series setters toggle the refresh flag
    @property
    def x(self): return self._x
    @x.setter
    def x(self, value): 
        self._x = value
        self._needs_refresh = True

    @property
    def y(self): return self._y
    @y.setter
    def y(self, value): 
        self._y = value
        self._needs_refresh = True

    @property
    def z(self): return self._z
    @z.setter
    def z(self, value): 
        self._z = value
        self._needs_refresh = True


    def _update_cache(self):
        """Refresh cached coefficient arrays (flattened lists)."""
        if not self._needs_refresh:
            return
        
        def extract(s):
            if s is None:
                return [], []
            coeffs = s.coeffs
            A = list(coeffs.values())
            n = list(coeffs.keys())
            return A, n

        self._Ax, self._nx = extract(self._x)
        self._Ay, self._ny = extract(self._y)
        self._Az, self._nz = extract(self._z)
        
        self.series_class   = _infer_class(self)
        self._dim           = _infer_dim(self)
        self._needs_refresh = False

    #===========================================================
    # Copying
    #===========================================================
    def copy(self,**overrides):
        """Deep copy of the Torus (series and parameters)."""
        return Torus(
            x=self._x.copy(**overrides) if self._x else None,
            y=self._y.copy(**overrides) if self._y else None,
            z=self._z.copy(**overrides) if self._z else None,
            bet0=(self.betx0, self.bety0, self.betz0)
        )
    
    #===========================================================
    # Dephasing
    #===========================================================
    def dephase(self,*args, **kwargs):
        """Dephase all projections."""
        return Torus(
            x=self._x.dephase(*args, **kwargs) if self._x else None,
            y=self._y.dephase(*args, **kwargs) if self._y else None,
            z=self._z.dephase(*args, **kwargs) if self._z else None,
            bet0=(self.betx0, self.bety0, self.betz0)
        )

    #===========================================================
    # Collapse
    #===========================================================
    def collapse(self,I, **kwargs):
        """Collapse all projections."""
        assert self.series_class is NormalFormSeries, "Collapse only defined for NormalFormSeries."
        return Torus(
            x=self._x.collapse(I, **kwargs) if self._x else None,
            y=self._y.collapse(I, **kwargs) if self._y else None,
            z=self._z.collapse(I, **kwargs) if self._z else None,
            bet0=(self.betx0, self.bety0, self.betz0)
        )

    #===========================================================
    # Evaluation on torus
    #===========================================================
    def eval(self,plane, Tx= 0, Ty=0, Tz=0):
        series = getattr(self, f"_{plane}")
        if series is None:
            raise ValueError(f"Torus has no '{plane}' plane defined.")
        return series.eval([Tx,Ty,Tz])
    

    # Coordinates evaluation
    #=====================================================================================
    def X(self,Tx=0,Ty=0,Tz=0):
        return np.real(self.eval('x',Tx,Ty,Tz))
    
    def PX(self,Tx=0,Ty=0,Tz=0):
        return -np.imag(self.eval('x',Tx,Ty,Tz))
    
    def Y(self,Tx=0,Ty=0,Tz=0):
        return np.real(self.eval('y',Tx,Ty,Tz))
    
    def PY(self,Tx=0,Ty=0,Tz=0):
        return -np.imag(self.eval('y',Tx,Ty,Tz))
    
    def Z(self,Tx=0,Ty=0,Tz=0):
        return np.real(self.eval('z',Tx,Ty,Tz))
    
    def PZ(self,Tx=0,Ty=0,Tz=0):
        return -np.imag(self.eval('z',Tx,Ty,Tz))
    #=====================================================================================






    #===========================================================
    # Actions
    #===========================================================
    def Ij(self,int_plane):
        self._update_cache()
        assert self.series_class is FourierSeries, "Ij integral only defined for FourierSeries."
        _Ijx = mathlib.poincare_avg(self._Ax, self._nx, int_plane, mp=self.mp)
        _Ijy = mathlib.poincare_avg(self._Ay, self._ny, int_plane, mp=self.mp) if self._y else 0
        _Ijz = mathlib.poincare_avg(self._Az, self._nz, int_plane, mp=self.mp) if self._z else 0
        return (_Ijx + _Ijy + _Ijz)/(2*self.mp.pi)

    @property
    def Ix(self): return self.Ij('x')
    
    @property
    def Iy(self): return self.Ij('y')

    @property
    def Iz(self): return self.Ij('z')

    @property
    def Jx(self):
        self._update_cache()
        return 0.5 * np.sum(np.abs(self._Ax)**2)
    
    @property
    def Jy(self):
        self._update_cache()
        return 0.5 * np.sum(np.abs(self._Ay)**2)
    
    @property
    def Jz(self):
        self._update_cache()
        return 0.5 * np.sum(np.abs(self._Az)**2)




    @property
    def R(self):
        """
        Nonlinear residual:
            R = sqrt( Σ (J_j - ⟨I_j⟩)² / Σ ⟨I_j⟩² )
        """
        self._update_cache()
        mp = self.mp
        if self.dim == 1:
            Ix, Iy, Iz = self.Ix, 0,0
            Jx, Jy, Jz = self.Jx, 0,0
        if self.dim == 2:
            Ix, Iy, Iz = self.Ix, self.Iy, 0
            Jx, Jy, Jz = self.Jx, self.Jy, 0
        if self.dim == 3:
            Ix, Iy, Iz = self.Ix, self.Iy, self.Iz
            Jx, Jy, Jz = self.Jx, self.Jy, self.Jz

        numerator   = (Jx - Ix)**2 + (Jy - Iy)**2 + (Jz - Iz)**2
        denominator = Ix**2 + Iy**2 + Iz**2
        return mp.sqrt(numerator / denominator) if denominator != 0 else mp.nan
    

    @property
    def R_T(self):
        """
        Transverse Non-linear residual:
            R = sqrt( Σ (J_j - ⟨I_j⟩)² / Σ ⟨I_j⟩² )
        """
        self._update_cache()
        mp = self.mp
        if self.dim == 1:
            Ix, Iy = self.Ix, 0
            Jx, Jy = self.Jx, 0
        else:
            Ix, Iy = self.Ix, self.Iy
            Jx, Jy = self.Jx, self.Jy

        numerator   = (Jx - Ix)**2 + (Jy - Iy)**2 
        denominator = Ix**2 + Iy**2
        return mp.sqrt(numerator / denominator) if denominator != 0 else mp.nan


    #===========================================================
    # Representation
    #===========================================================
    def __repr__(self):
        try:
            from pytori.display import render_torus
            return render_torus(self)
        except Exception:
            from pytori.display import render_torus_ascii
            return render_torus_ascii(self)