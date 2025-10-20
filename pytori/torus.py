import numpy as np
from shapely.geometry import Polygon
from shapely.plotting import patch_from_polygon
import pytori as pt
from collections import defaultdict



def _accumulate_modes(n_list, A_list, dim):
    """
    Helper to sum coefficients with the same aliased index n[:2]
    """
    acc = defaultdict(complex)
    for n, A in zip(n_list, A_list):
        acc[tuple(n[:dim])] += A
    return dict(acc)


class _TorusMeta(type):
    def __new__(cls, name, bases, namespace):
        def make_property(attr):
            # Getter function for properties
            def fget(self):
                if self.needs_refresh:
                    self.update_coeffs()
                return getattr(self, f"_{attr}")

            # Setter function for properties
            def fset(self, value):
                raise AttributeError(f"'{attr}' is a read-only property of Torus")

            return property(fget, fset)
        
        for attr in ["Ax", "Ay", "Az", "nx", "ny", "nz", "Nhx", "Nhy", "Nhz", "Nh", "dim", "betx0", "bety0", "betz0"]:
            namespace[attr] = make_property(attr)

        return super().__new__(cls, name, bases, namespace)
    

def _ndim_of(dct):
    return len(next(iter(dct))) if dct else 0


class Torus(metaclass=_TorusMeta):

    def __init__(self, x=None, y=None, z=None,betx0=1, bety0=1, betz0=1,
                 max_order=None, max_k=500, content_fraction=None,
                 numerical_tol=1e-21, mp='numpy', complex_basis=True,dim=None):
        


        # Initializing
        x = x or {}
        y = y or {}
        z = z or {}

        if dim is None:
            dim = max(_ndim_of(x), _ndim_of(y), _ndim_of(z))
        

        # Strong Fourier series
        self._x = pt.FourierSeriesND(x,dim=dim,max_order=max_order, max_k=max_k, content_fraction=content_fraction, numerical_tol=numerical_tol, bet0=betx0, mp=mp, complex_basis=complex_basis)
        self._y = pt.FourierSeriesND(y,dim=dim,max_order=max_order, max_k=max_k, content_fraction=content_fraction, numerical_tol=numerical_tol, bet0=bety0, mp=mp, complex_basis=complex_basis)
        self._z = pt.FourierSeriesND(z,dim=dim,max_order=max_order, max_k=max_k, content_fraction=content_fraction, numerical_tol=numerical_tol, bet0=betz0, mp=mp, complex_basis=complex_basis)
        
        # Updating coefficients and checks
        self.needs_refresh = True
        self.update_coeffs()
        self.make_checks()
        


    @classmethod
    def from_naff(cls,n=None,A=None, **kwargs):
        """
        Alternative constructor from (n, A) pairs, e.g., from NAFF output.
        Automatically handles aliasing by summing contributions with identical base frequency vectors.
        """
        assert len(A) == len(n), "A and n must have the same length"
        
        # Initialize
        x = y = z = None  
        dim = len(A)

        if dim >= 1:
            x = _accumulate_modes(n[0], A[0], dim)
        if dim >= 2:
            y = _accumulate_modes(n[1], A[1], dim)
        if dim >= 3:
            z = _accumulate_modes(n[2], A[2], dim)

        return cls(x=x, y=y, z=z,dim=dim, **kwargs)
    
    # We update coefficients if fourier series are modified
    #=======================================================
    @property
    def x(self): return self._x
    @x.setter
    def x(self, value): 
        self._x = value
        self.needs_refresh = True

    @property
    def y(self): return self._y
    @y.setter
    def y(self, value): 
        self._y = value
        self.needs_refresh = True

    @property
    def z(self): return self._z
    @z.setter
    def z(self, value): 
        self._z = value
        self.needs_refresh = True
    #=======================================================

    def update_coeffs(self):
        cx = self.x.to_dict()
        cy = self.y.to_dict()
        cz = self.z.to_dict()

        # Asigning values and forcing types
        #-------------
        self._Ax = [complex(A) for A in cx.values()]
        self._Ay = [complex(A) for A in cy.values()]
        self._Az = [complex(A) for A in cz.values()]
               
        self._nx = [tuple(int(_n) for _n in n) for n in cx.keys()]
        self._ny = [tuple(int(_n) for _n in n) for n in cy.keys()]
        self._nz = [tuple(int(_n) for _n in n) for n in cz.keys()]

        self._Nhx = len(self._Ax)
        self._Nhy = len(self._Ay)
        self._Nhz = len(self._Az)
        self._Nh  = max(self._Nhx, self._Nhy, self._Nhz)

        self._betx0 = self.x.bet0
        self._bety0 = self.y.bet0
        self._betz0 = self.z.bet0
        #-------------

        self._dim = len(self._nx[0]) if self._nx else 0

        # Closing the loop
        self.needs_refresh = False




    def make_checks(self):
        # Some checks as best as we can...
        #---------------------------------------------------------------------------------
        assert len(self.Ax) == len(self.nx), 'Ax and nx must have the same length'
        assert len(self.Ay) == len(self.ny), 'Ay and ny must have the same length'
        assert len(self.Az) == len(self.nz), 'Az and nz must have the same length'

        # assert len(self.nx) != 0, "nx must always be defined"
        
        # if self.dim == 1:
        #     assert len(self.ny) == 0, "ny must be empty in 2D"
        #     assert len(self.nz) == 0, "nz must be empty in 2D"

        # elif self.dim == 2:
        #     assert len(self.ny) != 0, "ny needs to be defined in 4D"
        #     assert len(self.nz) == 0, "nz must be empty in 4D"
        #     assert len(self.nx[0]) == len(self.ny[0]), 'nx and ny must have the same n. of dimensions'
        # elif self.dim == 3:
        #     assert len(self.ny) != 0, "ny needs to be defined in 6D"
        #     assert len(self.nz) != 0, "nz needs to be defined in 6D"
        #     assert len(self.nx[0]) == len(self.ny[0]), 'nx and ny must have the same n. of dimensions'
        #     assert len(self.nx[0]) == len(self.nz[0]), 'nx and nz must have the same n. of dimensions'
        #---------------------------------------------------------------------------------
    
    def copy(self):
        """Return a deep copy of this Torus."""
        _T = Torus(
            betx0=self.betx0,
            bety0=self.bety0,
            betz0=self.betz0,
            dim =self.dim,
            max_order=self.x.max_order,
            max_k=self.x.max_k,
            content_fraction=self.x.content_fraction,
            numerical_tol=self.x.numerical_tol,
            mp=self.x.mp,
            complex_basis=self.x.complex_basis,
        )
        # Replace Fourier series with deep copies
        _T.x = self.x.copy()
        _T.y = self.y.copy()
        _T.z = self.z.copy()

        # Recompute cached coefficients and checks for consistency
        _T.update_coeffs()
        _T.make_checks()
        return _T
        
    # Lambda evaluation
    def lambda_twiss(self,):
        """Evaluate the Lambda^+ and Lambda^- matrices (linear part!!) from the torus coefficients."""
        # Mathlib
        mp = self.x.mp

        # Basis vectors
        dim     = self.dim
        planes  = ['x','y','z'][:dim]
        basis   = [tuple(int(x) for x in np.eye(dim, dtype=int)[i]) for i in range(dim)]

        # Extracting eigen-vector-like quantities
        vecs = []
        for bp in basis: 
            # Negative basis
            bm = tuple(-x for x in bp)
            #-----
            Tkp = [getattr(self,plane)[bp] for plane in planes]                 # λ_k^+
            Tkm = [np.conjugate(getattr(self,plane)[bm]) for plane in planes]   # (λ_k^-)*
            #-----
            # [λ_k^+, (λ_k^-)*)]
            vecs.append(Tkp+Tkm)  


        # Normalizing and regauging
        vecs,norms  = pt.linear_normal_form._normalize_eigenvecs(vecs,mp=mp)
        vecs        = pt.linear_normal_form._regauge_eigenvecs(vecs,mp=mp)

        # Building Lambda^+, Lambda^-
        Lp,Lm = pt.linear_normal_form.T_to_lambda(vecs, mp=mp)
        return Lp,Lm

    # PSI Evaluation
    #=====================================================================================
    def _Psij(self,A,n,Tx=0,Ty=0,Tz=0,Nh=None):
        if Nh is None:
            Nh = len(A)
        else:
            Nh = int(Nh)

        if self.dim == 1:
            arg = [nk[0]*Tx for nk in n]
        elif self.dim == 2:
            arg = [nk[0]*Tx + nk[1]*Ty for nk in n]
        elif self.dim == 3:
            arg = [nk[0]*Tx + nk[1]*Ty + nk[2]*Tz for nk in n]
        else:
            raise ValueError('Invalid number of dimensions')
        
        _Psij = sum([Ak * np.exp(1j * argk)  for Ak,argk in zip(A[:Nh],arg[:Nh])])
        return _Psij
    

    def Psix(self,Tx=0,Ty=0,Tz=0,Nh=None,unpack = False):
        _Psij = self._Psij(self.Ax,self.nx,Tx=Tx,Ty=Ty,Tz=Tz,Nh=Nh)
        if unpack:
            return np.real(_Psij)*np.sqrt(self.betx0),-np.imag(_Psij)/np.sqrt(self.betx0)
        else:
            if self.betx0 == 1:
                return _Psij
            else: 
                return np.real(_Psij)*np.sqrt(self.betx0) - 1j * np.imag(_Psij)/np.sqrt(self.betx0)

    
    def Psiy(self,Tx=0,Ty=0,Tz=0,Nh=None,unpack = False):
        _Psij = self._Psij(self.Ay,self.ny,Tx=Tx,Ty=Ty,Tz=Tz,Nh=Nh)
        if unpack:
            return np.real(_Psij)*np.sqrt(self.bety0),-np.imag(_Psij)/np.sqrt(self.bety0)
        else:
            if self.betx0 == 1:
                return _Psij
            else:
                return np.real(_Psij)*np.sqrt(self.bety0) - 1j * np.imag(_Psij)/np.sqrt(self.bety0)
            
    def Psiz(self,Tx=0,Ty=0,Tz=0,Nh=None,unpack = False):
        _Psij = self._Psij(self.Az,self.nz,Tx=Tx,Ty=Ty,Tz=Tz,Nh=Nh)
        if unpack:
            return np.real(_Psij)*np.sqrt(self.betz0),-np.imag(_Psij)/np.sqrt(self.betz0)
        else:
            if self.betx0 == 1:
                return _Psij
            else:
                return np.real(_Psij)*np.sqrt(self.betz0) - 1j * np.imag(_Psij)/np.sqrt(self.betz0)
    #=====================================================================================



    # Coordinates evaluation
    #=====================================================================================
    def X(self,Tx=0,Ty=0,Tz=0,Nh=None):
        return np.real(self.Psix(Tx,Ty,Tz,Nh))
    
    def Px(self,Tx=0,Ty=0,Tz=0,Nh=None):
        return -np.imag(self.Psix(Tx,Ty,Tz,Nh))
    
    def Y(self,Tx=0,Ty=0,Tz=0,Nh=None):
        return np.real(self.Psiy(Tx,Ty,Tz,Nh))
    
    def Py(self,Tx=0,Ty=0,Tz=0,Nh=None):
        return -np.imag(self.Psiy(Tx,Ty,Tz,Nh))

    def Z(self,Tx=0,Ty=0,Tz=0,Nh=None):
        return np.real(self.Psiz(Tx,Ty,Tz,Nh))
    
    def Pz(self,Tx=0,Ty=0,Tz=0,Nh=None):
        return -np.imag(self.Psiz(Tx,Ty,Tz,Nh))    
    #=====================================================================================


    # Partial action evaluation
    #=====================================================================================
    def _phi(self,A,n,int_angle,Tx=0,Ty=0,Tz=0):
        if int_angle == 'x':
            if self.dim == 2:
                phi = [np.angle(Ak) + nk[1]*Ty  for Ak,nk in zip(A,n)]
            elif self.dim == 3:
                phi = [np.angle(Ak) + nk[1]*Ty + nk[2]*Tz  for Ak,nk in zip(A,n)]
        elif int_angle == 'y':
            if self.dim == 2:
                phi = [np.angle(Ak) + nk[0]*Tx  for Ak,nk in zip(A,n)]
            elif self.dim == 3:
                phi = [np.angle(Ak) + nk[0]*Tx + nk[2]*Tz  for Ak,nk in zip(A,n)]
        elif int_angle == 'z':
            if self.dim == 3:
                phi = [np.angle(Ak) + nk[0]*Tx + nk[1]*Ty  for Ak,nk in zip(A,n)]
        else:
            raise ValueError('Invalid integration angle')
        return phi

    def _Ijl(self,A,n,int_angle,Tx=0,Ty=0,Tz=0,Nh=None):
        # See paper, Eq. (D6)
        if Nh is None:
            Nh = len(A)
        else:
            Nh = int(Nh)

        phi     = self._phi(A,n,int_angle,Tx,Ty,Tz)
        jidx    = {'x':0,'y':1,'z':2}[int_angle]
        _Ijl    = 1/2 * sum(np.abs(Ak)*np.abs(Aj)*nk[jidx]*np.cos(phik-phij)    for nk,Ak,phik in zip(n[:Nh],A[:Nh],phi[:Nh]) 
                                                                                for nj,Aj,phij in zip(n[:Nh],A[:Nh],phi[:Nh]) 
                                                                                if nj[jidx]==nk[jidx])
        
        return _Ijl
    
    # Integration on Theta-X
    def Ixx(self,Ty=0,Tz=0,Nh=None):
        return self._Ijl(self.Ax,self.nx,int_angle='x',Ty=Ty,Tz=Tz,Nh=Nh)
    def Ixy(self,Ty=0,Tz=0,Nh=None):
        return self._Ijl(self.Ay,self.ny,int_angle='x',Ty=Ty,Tz=Tz,Nh=Nh)
    def Ixz(self,Ty=0,Tz=0,Nh=None):
        return self._Ijl(self.Az,self.nz,int_angle='x',Ty=Ty,Tz=Tz,Nh=Nh)
    
    # Integration on Theta-Y
    def Iyx(self,Tx=0,Tz=0,Nh=None):
        return self._Ijl(self.Ax,self.nx,int_angle='y',Tx=Tx,Tz=Tz,Nh=Nh)
    def Iyy(self,Tx=0,Tz=0,Nh=None):
        return self._Ijl(self.Ay,self.ny,int_angle='y',Tx=Tx,Tz=Tz,Nh=Nh)
    def Iyz(self,Tx=0,Tz=0,Nh=None):
        return self._Ijl(self.Az,self.nz,int_angle='y',Tx=Tx,Tz=Tz,Nh=Nh)
    
    # Integration on Theta-Z
    def Izx(self,Tx=0,Ty=0,Nh=None):
        return self._Ijl(self.Ax,self.nx,int_angle='z',Tx=Tx,Ty=Ty,Nh=Nh)
    def Izy(self,Tx=0,Ty=0,Nh=None):
        return self._Ijl(self.Ay,self.ny,int_angle='z',Tx=Tx,Ty=Ty,Nh=Nh)
    def Izz(self,Tx=0,Ty=0,Nh=None):
        return self._Ijl(self.Az,self.nz,int_angle='z',Tx=Tx,Ty=Ty,Nh=Nh)
    #=====================================================================================

    
    # Delta function evaluation
    #=====================================================================================
    def _Djl(self,A,n,int_angle,Tx=0,Ty=0,Tz=0,Nh=None):
        # CHANGED. Dij now refers to the area-delta, not the action. See Thesis.
        if Nh is None:
            Nh = len(A)
        else:
            Nh = int(Nh)

        phi     = self._phi(A,n,int_angle,Tx,Ty,Tz)
        jidx    = {'x':0,'y':1,'z':2}[int_angle]
        _Djl    = np.pi * sum(np.abs(Ak)*np.abs(Aj)*nk[jidx]*np.cos(phik-phij)    for k,nk,Ak,phik in zip(range(Nh),n[:Nh],A[:Nh],phi[:Nh]) 
                                                                                for j,nj,Aj,phij in zip(range(Nh),n[:Nh],A[:Nh],phi[:Nh]) 
                                                                                if (nj[jidx]==nk[jidx]) and (j!=k))
        
        return _Djl

    # Integration on Theta-X
    def Dxx(self,Ty=0,Tz=0,Nh=None):
        return self._Djl(self.Ax,self.nx,int_angle='x',Ty=Ty,Tz=Tz,Nh=Nh)
    def Dxy(self,Ty=0,Tz=0,Nh=None):
        return self._Djl(self.Ay,self.ny,int_angle='x',Ty=Ty,Tz=Tz,Nh=Nh)
    def Dxz(self,Ty=0,Tz=0,Nh=None):
        return self._Djl(self.Az,self.nz,int_angle='x',Ty=Ty,Tz=Tz,Nh=Nh)
    
    # Integration on Theta-Y
    def Dyx(self,Tx=0,Tz=0,Nh=None):
        return self._Djl(self.Ax,self.nx,int_angle='y',Tx=Tx,Tz=Tz,Nh=Nh)
    def Dyy(self,Tx=0,Tz=0,Nh=None):
        return self._Djl(self.Ay,self.ny,int_angle='y',Tx=Tx,Tz=Tz,Nh=Nh)
    def Dyz(self,Tx=0,Tz=0,Nh=None):
        return self._Djl(self.Az,self.nz,int_angle='y',Tx=Tx,Tz=Tz,Nh=Nh)
    
    # Integration on Theta-Z
    def Dzx(self,Tx=0,Ty=0,Nh=None):
        return self._Djl(self.Ax,self.nx,int_angle='z',Tx=Tx,Ty=Ty,Nh=Nh)
    def Dzy(self,Tx=0,Ty=0,Nh=None):
        return self._Djl(self.Ay,self.ny,int_angle='z',Tx=Tx,Ty=Ty,Nh=Nh)
    def Dzz(self,Tx=0,Ty=0,Nh=None):
        return self._Djl(self.Az,self.nz,int_angle='z',Tx=Tx,Ty=Ty,Nh=Nh)
    #=====================================================================================

    
    # Invariant evaluation
    #=====================================================================================
    def Ix(self,Ty=0,Tz=0,Nh=None):
        return self.Ixx(Ty,Tz,Nh) + self.Ixy(Ty,Tz,Nh) + self.Ixz(Ty,Tz,Nh)
    def Iy(self,Tx=0,Tz=0,Nh=None):
        return self.Iyx(Tx,Tz,Nh) + self.Iyy(Tx,Tz,Nh) + self.Iyz(Tx,Tz,Nh)
    def Iz(self,Tx=0,Ty=0,Nh=None):
        return self.Izx(Tx,Ty,Nh) + self.Izy(Tx,Ty,Nh) + self.Izz(Tx,Ty,Nh)
    #-------------------------------------
    def epsx(self,Ty=0,Tz=0,Nh=None):
        return self.Dxx(Ty,Tz,Nh) + self.Dxy(Ty,Tz,Nh) + self.Dxz(Ty,Tz,Nh)
    def epsy(self,Tx=0,Tz=0,Nh=None):
        return self.Dyx(Tx,Tz,Nh) + self.Dyy(Tx,Tz,Nh) + self.Dyz(Tx,Tz,Nh)
    def epsz(self,Tx=0,Ty=0,Nh=None):
        return self.Dzx(Tx,Ty,Nh) + self.Dzy(Tx,Ty,Nh) + self.Dzz(Tx,Ty,Nh)
    #=====================================================================================


    # AVG Invariant evaluation
    #=====================================================================================
    def _EIj(self,int_angle,Nh = None,Nhx=None,Nhy=None,Nhz=None):
        if Nhx is None:
            Nhx = len(self.Ax)
        if Nhy is None:
            Nhy = len(self.Ay)
        if Nhz is None:
            Nhz = len(self.Az)
        if Nh is None:
            pass
        else:
            Nhx = int(np.min([Nh,Nhx]))
            Nhy = int(np.min([Nh,Nhy]))
            Nhz = int(np.min([Nh,Nhz]))
        
        jidx  = {'x':0,'y':1,'z':2}[int_angle]
        _EIjx = 1/2 * sum([nk[jidx]*(np.abs(Ak)**2) for nk,Ak in zip(self.nx[:Nhx],self.Ax[:Nhx])])
        _EIjy = 1/2 * sum([nk[jidx]*(np.abs(Ak)**2) for nk,Ak in zip(self.ny[:Nhy],self.Ay[:Nhy])])
        _EIjz = 1/2 * sum([nk[jidx]*(np.abs(Ak)**2) for nk,Ak in zip(self.nz[:Nhz],self.Az[:Nhz])])
        
        return _EIjx+_EIjy+_EIjz
    
    @property
    def EIx(self):
        return self._EIj('x')
    @property
    def EIy(self):
        return self._EIj('y')
    @property
    def EIz(self):
        return self._EIj('z')
    
    def EIx_truncate(self,Nh = None,Nhx=None,Nhy=None,Nhz=None):
        return self._EIj('x',Nh,Nhx,Nhy,Nhz)
    def EIy_truncate(self,Nh = None,Nhx=None,Nhy=None,Nhz=None):
        return self._EIj('y',Nh,Nhx,Nhy,Nhz)
    def EIz_truncate(self,Nh = None,Nhx=None,Nhy=None,Nhz=None):
        return self._EIj('z',Nh,Nhx,Nhy,Nhz)
    #=====================================================================================


    # Courant-snyder invariant, x^2 + px^2
    #=====================================================================================
    def _Jj(self, A ,Nh = None):
        if Nh is None:
            Nh = len(A)
        else:
            Nh = int(Nh)

        return 1/2 * sum([(np.abs(Ak)**2) for Ak in A[:Nh]])
    
    @property
    def Jx(self):
        return self._Jj(self.Ax)
    @property
    def Jy(self):
        return self._Jj(self.Ay)
    @property
    def Jz(self):
        return self._Jj(self.Az)
    #=====================================================================================

    # Non-linear residual
    @property
    def R(self):
        
        err = []
        actions = []
        for plane in ['x','y','z'][:self.dim]:
            _J = getattr(self,f'J{plane}')
            _I = self._EIj(f'{plane}')
            err.append(_J-_I)
            actions.append(_I)
        
        R2 = sum(e**2 for e in err)/sum(a**2 for a in actions)
        return np.sqrt(R2)

        

            

        



    # Plotting toolbox
    #=====================================================================================
    def loop(self, Tx=0, Ty=0, Tz=0, Nh=None, partial=False):
        if partial:
            addon = [(0, 0)]
        else:
            addon = []

        projections = []  # use list for accumulation

        if self.dim >= 1:
            X, Px = self.Psix(Tx=Tx, Ty=Ty, Tz=Tz, Nh=Nh, unpack=True)
            loopx = Polygon(addon + list(zip(X, Px)))
            projections.append(loopx)

        if self.dim >= 2:
            Y, Py = self.Psiy(Tx=Tx, Ty=Ty, Tz=Tz, Nh=Nh, unpack=True)
            loopy = Polygon(addon + list(zip(Y, Py)))
            projections.append(loopy)

        if self.dim >= 3:
            Z, Pz = self.Psiz(Tx=Tx, Ty=Ty, Tz=Tz, Nh=Nh, unpack=True)
            loopz = Polygon(addon + list(zip(Z, Pz)))
            projections.append(loopz)

        return tuple(projections)  # return as tuple for immutability

        
    
    def loop_patch(self, Tx=0, Ty=0, Tz=0, Nh=None, partial=False, unpack=False, **kwargs):
        loop_projs = self.loop(Tx=Tx, Ty=Ty, Tz=Tz, Nh=Nh, partial=partial)
        patches = [patch_from_polygon(loop, **kwargs) for loop in loop_projs]

        if unpack:
            return loop_projs, patches
        else:
            return patches
    #=====================================================================================




class Mesh():
    def __init__(self,r,verts,faces,verts_in=None,faces_in=None,polycenter = [0,0,0]):
        self.r = r
        self.verts_in = verts_in
        self.faces_in = faces_in
        self.edges_in = []
        self.verts = verts
        self.faces = faces
        self.edges = []
        self.meta = {}
        self.polycenter = np.array(polycenter)

    @classmethod
    def from_torus(cls,torus,plane_ij='xx',slice_along='y',r = None,r_rescale = 1,num_angles=100,num_slices=100,theta_angle=None,theta_slice=None,Tx=0, Ty=0, Tz=0,inner_scale=None):
        
        assert plane_ij in ['xx','xy','xz',
                            'yx','yy','yz',
                            'zx','zy','zz',], 'plane_ij must be one of xy, xz, yz'

        assert slice_along != plane_ij[0], 'slice_along must be different from the integration angle (first index of plane_ij)'
        

        if theta_angle is None:
            theta_angle     = np.linspace(0,2*np.pi,num_angles)
            close_angle     = True
        else:
            num_angles = len(theta_angle)
            close_angle = False


        if theta_slice is None:
            theta_slice = np.linspace(0,2*np.pi,num_slices)
            close_slice = True
        else:
            num_slices = len(theta_slice)
            close_slice = False
        


        # Choosing proper projection
        projection = {'x':torus.Psix,
                      'y':torus.Psiy,
                      'z':torus.Psiz}[plane_ij[1]]


        # Preparing argument dictionnary
        angle_dict = {'Tx':Tx, 'Ty':Ty, 'Tz':Tz,'unpack':True}
        angle_dict.update({f'T{plane_ij[0]}': theta_angle})

        if r is None:
            r = np.sqrt(2*torus._EIj(plane_ij[0]))
        r *= r_rescale

        # Generating the slices
        slices = []
        slices_in = []
        slice_offset = angle_dict[f'T{slice_along}']
        for t0 in theta_slice:
            # Coordinates
            angle_dict.update({f'T{slice_along}': t0+slice_offset})
            x,z = projection(**angle_dict)
            
            # Recentering
            center_x = r*np.cos(t0)
            center_y = r*np.sin(t0)
            center_z = 0
            
            slices.append([ center_x + x*np.cos(t0),
                            center_y + x*np.sin(t0),
                            center_z + z])

            if inner_scale is not None:
                slices_in.append([  center_x + inner_scale*x*np.cos(t0),
                                    center_y + inner_scale*x*np.sin(t0),
                                    center_z + inner_scale*z])



        # Building the torus mesh
        #---------------------------------------------------------------------------------
        #===================
        #   s,i+1       s+1,i+1  
        #    +---------+  
        #    |         |  
        #    |    F    |  
        #    |         |  
        #    +---------+  
        #   s,i        s+1,i  
        #===================
        # OUTER SURFACE
        _xyz        = np.array(slices).transpose(0, 2, 1)
        v_idx_out   = np.arange(_xyz.shape[0]*_xyz.shape[1]).reshape((_xyz.shape[0],_xyz.shape[1]))
        

        if close_slice:
            s_range = range(-1, num_slices - 1)
        else:
            s_range = range(num_slices - 1)
            
        if close_angle:
            i_range = range(-1, num_angles - 1)
        else:
            i_range = range(num_angles - 1)

        verts_out   = _xyz.reshape(-1, _xyz.shape[-1]).tolist()
        faces_out   =[[ v_idx_out[s  ,i  ],
                        v_idx_out[s  ,i+1],
                        v_idx_out[s+1,i+1],
                        v_idx_out[s+1,i  ]]  for s in s_range for i in i_range]
        
        if inner_scale is not None:
            # INNER SURFACE
            _xyz    = np.array(slices_in).transpose(0, 2, 1)
            v_idx_in= np.arange(_xyz.shape[0]*_xyz.shape[1]).reshape((_xyz.shape[0],_xyz.shape[1]))
            
            verts_in= _xyz.reshape(-1, _xyz.shape[-1]).tolist()
            faces_in=[[ v_idx_in[s  ,i  ],
                        v_idx_in[s+1,i  ],
                        v_idx_in[s+1,i+1],
                        v_idx_in[s  ,i+1]]  for s in s_range for i in i_range]
        else: 
            verts_in=None
            faces_in=None


        return cls(r,verts_out,faces_out,verts_in,faces_in)


    def to_Poly3DCollection(self,edgecolor='none',linewidths=0,alpha=1,rasterized=True,**kwargs):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        kwargs.update({'edgecolor': edgecolor, 'linewidths': linewidths, 'alpha': alpha})
        collection = Poly3DCollection(self.poly3d, **kwargs)
        if rasterized:
            collection.set_rasterized(True)     # Needed for complex lighting in PDF
            collection.set_antialiaseds(False)  # Smooth edges
            if edgecolor == 'none':
                collection.set_edgecolor((0,0,0,0)) # Fully transparent edges
        return collection
    
    @property
    def poly3d(self):
        shift = self.polycenter
        return [[(np.array(self.verts[vert_id]) + shift).tolist() for vert_id in face] for face in self.faces]

    @property
    def scale(self):
        all_verts = np.array([v for face in self.poly3d for v in face])
        return all_verts.flatten()


    def to_dict(self):
        return self.__dict__.copy()
    
    def to_pickle(self,filename):
        import pickle

        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    def to_json(self,filename):
        metadata = self.to_dict()
        with open(filename , "w") as f: 
            json.dump(metadata, f,cls=NpEncoder)



        
#============================================================
import json
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
#============================================================




class Torus_old():
    def __init__(self,A=None,n=None,Ax=[],Ay=[],Az=[],nx=[],ny=[],nz=[],betx0=1,bety0=1,betz0=1):
        
        if A is not None:
            assert len(Ax)==0 and len(Ay)==0 and len(Az)==0, 'Ax,Ay,Az must be empty if A is defined'
            if len(A)==1:
                Ax, = A
            elif len(A)==2:
                Ax,Ay = A
            elif len(A)==3:
                Ax,Ay,Az = A
        if n is not None:
            assert len(nx)==0 and len(ny)==0 and len(nz)==0, 'nx,ny,nz must be empty if n is defined'
            if len(n)==1:
                nx, = n
            elif len(n)==2:
                nx,ny = n
            elif len(n)==3:
                nx,ny,nz = n

    
        # Asigning values and forcing types
        #-------------
        self.Ax = [complex(A) for A in Ax]
        self.Ay = [complex(A) for A in Ay]
        self.Az = [complex(A) for A in Az]
        
        # Dropping aliasing term
        self.nx = [tuple(int(_n) for _n in n) for n in nx]
        self.ny = [tuple(int(_n) for _n in n) for n in ny]
        self.nz = [tuple(int(_n) for _n in n) for n in nz]

        self.Nhx = len(self.Ax)
        self.Nhy = len(self.Ay)
        self.Nhz = len(self.Az)
        self.Nh  = np.max([self.Nhx,self.Nhy,self.Nhz])

        self.betx0 = betx0
        self.bety0 = bety0
        self.betz0 = betz0
        #-------------
        

        # Some checks as best as we can...
        #---------------------------------------------------------------------------------
        assert len(Ax) == len(nx), 'Ax and nx must have the same length'
        assert len(Ay) == len(ny), 'Ay and ny must have the same length'
        assert len(Az) == len(nz), 'Az and nz must have the same length'

        assert len(nx) != 0, "nx must always be defined"
        self.dim = len(self.nx[0])-1
        if self.dim == 1:
            assert len(ny) == 0, "ny must be empty in 2D"
            assert len(nz) == 0, "nz must be empty in 2D"

        elif self.dim == 2:
            assert len(ny) != 0, "ny needs to be defined in 4D"
            assert len(nz) == 0, "nz must be empty in 4D"
            assert len(nx[0]) == len(ny[0]), 'nx and ny must have the same n. of dimensions'
        elif self.dim == 3:
            assert len(ny) != 0, "ny needs to be defined in 6D"
            assert len(nz) != 0, "nz needs to be defined in 6D"
            assert len(nx[0]) == len(ny[0]), 'nx and ny must have the same n. of dimensions'
            assert len(nx[0]) == len(nz[0]), 'nx and nz must have the same n. of dimensions'
        #---------------------------------------------------------------------------------



    # PSI Evaluation
    #=====================================================================================
    def _Psij(self,A,n,Tx=0,Ty=0,Tz=0,Nh=None):
        if Nh is None:
            Nh = len(A)
        else:
            Nh = int(Nh)

        if self.dim == 1:
            arg = [nk[0]*Tx for nk in n]
        elif self.dim == 2:
            arg = [nk[0]*Tx + nk[1]*Ty for nk in n]
        elif self.dim == 3:
            arg = [nk[0]*Tx + nk[1]*Ty + nk[2]*Tz for nk in n]
        else:
            raise ValueError('Invalid number of dimensions')
        
        _Psij = sum([Ak * np.exp(1j * argk)  for Ak,argk in zip(A[:Nh],arg[:Nh])])
        return _Psij
    

    def Psix(self,Tx=0,Ty=0,Tz=0,Nh=None,unpack = False):
        _Psij = self._Psij(self.Ax,self.nx,Tx=Tx,Ty=Ty,Tz=Tz,Nh=Nh)
        if unpack:
            return np.real(_Psij),-np.imag(_Psij)
        else:
            return _Psij
    
    def Psiy(self,Tx=0,Ty=0,Tz=0,Nh=None,unpack = False):
        _Psij = self._Psij(self.Ay,self.ny,Tx=Tx,Ty=Ty,Tz=Tz,Nh=Nh)
        if unpack:
            return np.real(_Psij),-np.imag(_Psij)
        else:
            return _Psij
    
    def Psiz(self,Tx=0,Ty=0,Tz=0,Nh=None,unpack = False):
        _Psij = self._Psij(self.Az,self.nz,Tx=Tx,Ty=Ty,Tz=Tz,Nh=Nh)
        if unpack:
            return np.real(_Psij),-np.imag(_Psij)
        else:
            return _Psij
    #=====================================================================================



    # Coordinates evaluation
    #=====================================================================================
    def X(self,Tx=0,Ty=0,Tz=0,Nh=None):
        return np.real(self.Psix(Tx,Ty,Tz,Nh))
    
    def Px(self,Tx=0,Ty=0,Tz=0,Nh=None):
        return -np.imag(self.Psix(Tx,Ty,Tz,Nh))
    
    def Y(self,Tx=0,Ty=0,Tz=0,Nh=None):
        return np.real(self.Psiy(Tx,Ty,Tz,Nh))
    
    def Py(self,Tx=0,Ty=0,Tz=0,Nh=None):
        return -np.imag(self.Psiy(Tx,Ty,Tz,Nh))

    def Z(self,Tx=0,Ty=0,Tz=0,Nh=None):
        return np.real(self.Psiz(Tx,Ty,Tz,Nh))
    
    def Pz(self,Tx=0,Ty=0,Tz=0,Nh=None):
        return -np.imag(self.Psiz(Tx,Ty,Tz,Nh))    
    #=====================================================================================


    # Partial action evaluation
    #=====================================================================================
    def _phi(self,A,n,int_angle,Tx=0,Ty=0,Tz=0):
        if int_angle == 'x':
            if self.dim == 2:
                phi = [np.angle(Ak) + nk[1]*Ty  for Ak,nk in zip(A,n)]
            elif self.dim == 3:
                phi = [np.angle(Ak) + nk[1]*Ty + nk[2]*Tz  for Ak,nk in zip(A,n)]
        elif int_angle == 'y':
            if self.dim == 2:
                phi = [np.angle(Ak) + nk[0]*Tx  for Ak,nk in zip(A,n)]
            elif self.dim == 3:
                phi = [np.angle(Ak) + nk[0]*Tx + nk[2]*Tz  for Ak,nk in zip(A,n)]
        elif int_angle == 'z':
            if self.dim == 3:
                phi = [np.angle(Ak) + nk[0]*Tx + nk[1]*Ty  for Ak,nk in zip(A,n)]
        else:
            raise ValueError('Invalid integration angle')
        return phi

    def _Ijl(self,A,n,int_angle,Tx=0,Ty=0,Tz=0,Nh=None):
        # See paper, Eq. (D6)
        if Nh is None:
            Nh = len(A)
        else:
            Nh = int(Nh)

        phi     = self._phi(A,n,int_angle,Tx,Ty,Tz)
        jidx    = {'x':0,'y':1,'z':2}[int_angle]
        _Ijl    = 1/2 * sum(np.abs(Ak)*np.abs(Aj)*nk[jidx]*np.cos(phik-phij)    for nk,Ak,phik in zip(n[:Nh],A[:Nh],phi[:Nh]) 
                                                                                for nj,Aj,phij in zip(n[:Nh],A[:Nh],phi[:Nh]) 
                                                                                if nj[jidx]==nk[jidx])
        
        return _Ijl
    
    # Integration on Theta-X
    def Ixx(self,Ty=0,Tz=0,Nh=None):
        return self._Ijl(self.Ax,self.nx,int_angle='x',Ty=Ty,Tz=Tz,Nh=Nh)
    def Ixy(self,Ty=0,Tz=0,Nh=None):
        return self._Ijl(self.Ay,self.ny,int_angle='x',Ty=Ty,Tz=Tz,Nh=Nh)
    def Ixz(self,Ty=0,Tz=0,Nh=None):
        return self._Ijl(self.Az,self.nz,int_angle='x',Ty=Ty,Tz=Tz,Nh=Nh)
    
    # Integration on Theta-Y
    def Iyx(self,Tx=0,Tz=0,Nh=None):
        return self._Ijl(self.Ax,self.nx,int_angle='y',Tx=Tx,Tz=Tz,Nh=Nh)
    def Iyy(self,Tx=0,Tz=0,Nh=None):
        return self._Ijl(self.Ay,self.ny,int_angle='y',Tx=Tx,Tz=Tz,Nh=Nh)
    def Iyz(self,Tx=0,Tz=0,Nh=None):
        return self._Ijl(self.Az,self.nz,int_angle='y',Tx=Tx,Tz=Tz,Nh=Nh)
    
    # Integration on Theta-Z
    def Izx(self,Tx=0,Ty=0,Nh=None):
        return self._Ijl(self.Ax,self.nx,int_angle='z',Tx=Tx,Ty=Ty,Nh=Nh)
    def Izy(self,Tx=0,Ty=0,Nh=None):
        return self._Ijl(self.Ay,self.ny,int_angle='z',Tx=Tx,Ty=Ty,Nh=Nh)
    def Izz(self,Tx=0,Ty=0,Nh=None):
        return self._Ijl(self.Az,self.nz,int_angle='z',Tx=Tx,Ty=Ty,Nh=Nh)
    #=====================================================================================

    
    # Delta function evaluation
    #=====================================================================================
    def _Djl(self,A,n,int_angle,Tx=0,Ty=0,Tz=0,Nh=None):
        # See paper, Eq. (D8)
        if Nh is None:
            Nh = len(A)
        else:
            Nh = int(Nh)

        phi     = self._phi(A,n,int_angle,Tx,Ty,Tz)
        jidx    = {'x':0,'y':1,'z':2}[int_angle]
        _Djl    = 1/2 * sum(np.abs(Ak)*np.abs(Aj)*nk[jidx]*np.cos(phik-phij)    for k,nk,Ak,phik in zip(range(Nh),n[:Nh],A[:Nh],phi[:Nh]) 
                                                                                for j,nj,Aj,phij in zip(range(Nh),n[:Nh],A[:Nh],phi[:Nh]) 
                                                                                if (nj[jidx]==nk[jidx]) and (j!=k))
        
        return _Djl

    # Integration on Theta-X
    def Dxx(self,Ty=0,Tz=0,Nh=None):
        return self._Djl(self.Ax,self.nx,int_angle='x',Ty=Ty,Tz=Tz,Nh=Nh)
    def Dxy(self,Ty=0,Tz=0,Nh=None):
        return self._Djl(self.Ay,self.ny,int_angle='x',Ty=Ty,Tz=Tz,Nh=Nh)
    def Dxz(self,Ty=0,Tz=0,Nh=None):
        return self._Djl(self.Az,self.nz,int_angle='x',Ty=Ty,Tz=Tz,Nh=Nh)
    
    # Integration on Theta-Y
    def Dyx(self,Tx=0,Tz=0,Nh=None):
        return self._Djl(self.Ax,self.nx,int_angle='y',Tx=Tx,Tz=Tz,Nh=Nh)
    def Dyy(self,Tx=0,Tz=0,Nh=None):
        return self._Djl(self.Ay,self.ny,int_angle='y',Tx=Tx,Tz=Tz,Nh=Nh)
    def Dyz(self,Tx=0,Tz=0,Nh=None):
        return self._Djl(self.Az,self.nz,int_angle='y',Tx=Tx,Tz=Tz,Nh=Nh)
    
    # Integration on Theta-Z
    def Dzx(self,Tx=0,Ty=0,Nh=None):
        return self._Djl(self.Ax,self.nx,int_angle='z',Tx=Tx,Ty=Ty,Nh=Nh)
    def Dzy(self,Tx=0,Ty=0,Nh=None):
        return self._Djl(self.Ay,self.ny,int_angle='z',Tx=Tx,Ty=Ty,Nh=Nh)
    def Dzz(self,Tx=0,Ty=0,Nh=None):
        return self._Djl(self.Az,self.nz,int_angle='z',Tx=Tx,Ty=Ty,Nh=Nh)
    #=====================================================================================

    
    # Invariant evaluation
    #=====================================================================================
    def Ix(self,Ty=0,Tz=0,Nh=None):
        return self.Ixx(Ty,Tz,Nh) + self.Ixy(Ty,Tz,Nh) + self.Ixz(Ty,Tz,Nh)
    def Iy(self,Tx=0,Tz=0,Nh=None):
        return self.Iyx(Tx,Tz,Nh) + self.Iyy(Tx,Tz,Nh) + self.Iyz(Tx,Tz,Nh)
    def Iz(self,Tx=0,Ty=0,Nh=None):
        return self.Izx(Tx,Ty,Nh) + self.Izy(Tx,Ty,Nh) + self.Izz(Tx,Ty,Nh)
    #-------------------------------------
    def epsx(self,Ty=0,Tz=0,Nh=None):
        return self.Dxx(Ty,Tz,Nh) + self.Dxy(Ty,Tz,Nh) + self.Dxz(Ty,Tz,Nh)
    def epsy(self,Tx=0,Tz=0,Nh=None):
        return self.Dyx(Tx,Tz,Nh) + self.Dyy(Tx,Tz,Nh) + self.Dyz(Tx,Tz,Nh)
    def epsz(self,Tx=0,Ty=0,Nh=None):
        return self.Dzx(Tx,Ty,Nh) + self.Dzy(Tx,Ty,Nh) + self.Dzz(Tx,Ty,Nh)
    #=====================================================================================


    # AVG Invariant evaluation
    #=====================================================================================
    def _EIj(self,int_angle,Nh = None,Nhx=None,Nhy=None,Nhz=None):
        if Nhx is None:
            Nhx = len(self.Ax)
        if Nhy is None:
            Nhy = len(self.Ay)
        if Nhz is None:
            Nhz = len(self.Az)
        if Nh is None:
            pass
        else:
            Nhx = int(np.min([Nh,Nhx]))
            Nhy = int(np.min([Nh,Nhy]))
            Nhz = int(np.min([Nh,Nhz]))
        
        jidx  = {'x':0,'y':1,'z':2}[int_angle]
        _EIjx = 1/2 * sum([nk[jidx]*(np.abs(Ak)**2) for nk,Ak in zip(self.nx[:Nhx],self.Ax[:Nhx])])
        _EIjy = 1/2 * sum([nk[jidx]*(np.abs(Ak)**2) for nk,Ak in zip(self.ny[:Nhy],self.Ay[:Nhy])])
        _EIjz = 1/2 * sum([nk[jidx]*(np.abs(Ak)**2) for nk,Ak in zip(self.nz[:Nhz],self.Az[:Nhz])])
        
        return _EIjx+_EIjy+_EIjz
    
    @property
    def EIx(self):
        return self._EIj('x')
    @property
    def EIy(self):
        return self._EIj('y')
    @property
    def EIz(self):
        return self._EIj('z')
    
    def EIx_truncate(self,Nh = None,Nhx=None,Nhy=None,Nhz=None):
        return self._EIj('x',Nh,Nhx,Nhy,Nhz)
    def EIy_truncate(self,Nh = None,Nhx=None,Nhy=None,Nhz=None):
        return self._EIj('y',Nh,Nhx,Nhy,Nhz)
    def EIz_truncate(self,Nh = None,Nhx=None,Nhy=None,Nhz=None):
        return self._EIj('z',Nh,Nhx,Nhy,Nhz)
    #=====================================================================================


    # Courant-snyder invariant, x^2 + px^2
    #=====================================================================================
    def _Jj(self, A ,Nh = None):
        if Nh is None:
            Nh = len(A)
        else:
            Nh = int(Nh)

        return 1/2 * sum([(np.abs(Ak)**2) for Ak in A[:Nh]])
    
    @property
    def Jx(self,Nh = None):
        return self._Jj(self.Ax,Nh)
    @property
    def Jy(self,Nh = None):
        return self._Jj(self.Ay,Nh)
    @property
    def Jz(self,Nh = None):
        return self._Jj(self.Az,Nh)
    #=====================================================================================



    # Plotting toolbox
    #=====================================================================================
    def loop(self,Tx=0,Ty=0,Tz=0,Nh=None,partial = False):
        if partial:
            addon = [(0,0)]
        else:
            addon = []

        if self.dim>=1:
            X,Px = self.Psix(Tx=Tx,Ty=Ty,Tz=Tz,Nh=Nh,unpack = True)
            loopx = Polygon(addon + [(_x,_px) for _x,_px in zip(X,Px)])
            projections = loopx
        if self.dim>=2:
            Y,Py = self.Psiy(Tx=Tx,Ty=Ty,Tz=Tz,Nh=Nh,unpack = True)
            loopy = Polygon(addon + [(_y,_py) for _y,_py in zip(Y,Py)])
            projections += (loopy,)
        if self.dim>=3:
            Z,Pz = self.Psiz(Tx=Tx,Ty=Ty,Tz=Tz,Nh=Nh,unpack = True)
            loopz = Polygon(addon + [(_z,_pz) for _z,_pz in zip(Z,Pz)])
            projections += (loopz,)

        return projections
        
    
    def loop_patch(self,Tx=0,Ty=0,Tz=0,Nh=None,partial = False,unpack=False,**kwargs):
        _loop = self.loop(Tx=Tx,Ty=Ty,Tz=Tz,Nh=Nh,partial=partial)
        if unpack:
            return _loop,patch_from_polygon(_loop,**kwargs)
        else:
            return patch_from_polygon(_loop,**kwargs)
    #=====================================================================================