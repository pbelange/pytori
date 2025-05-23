import numpy as np
from shapely.geometry import Polygon
from shapely.plotting import patch_from_polygon


class Torus():
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