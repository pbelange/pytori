import numpy as np


class Torus():
    def __init__(self,Ax=None,Ay=None,Az=None,nx=None,ny=None,nz=None):
        self.Ax = Ax
        self.Ay = Ay
        self.Az = Az
        self.nx = nx
        self.ny = ny
        self.nz = nz

        self._Ix = None
        self._Iy = None
        self._Iz = None


    
    @property
    def Ix(self):
        if self._Ix is None:
            _ix = sum([n[0]*(np.abs(A)**2)/2 for n,A in zip(self.nx,self.Ax)])
            _iy = sum([n[0]*(np.abs(A)**2)/2 for n,A in zip(self.ny,self.Ay)])
            self._Ix = _ix + _iy
        return self._Ix
    
    
    @property
    def Iy(self):
        if self._Iy is None:
            _ix = sum([n[1]*(np.abs(A)**2)/2 for n,A in zip(self.nx,self.Ax)])
            _iy = sum([n[1]*(np.abs(A)**2)/2 for n,A in zip(self.ny,self.Ay)])
            self._Iy = _ix + _iy
        return self._Iy
    
    @property
    def Jx(self):
        _ix = sum([(np.abs(A)**2)/2 for n,A in zip(self.nx,self.Ax)])
        return _ix
    
    @property
    def Jy(self):
        _ix = sum([(np.abs(A)**2)/2 for n,A in zip(self.ny,self.Ay)])
        return _ix
    

    def draw_path(self,plane,Qx,Qy,N,r_aspect = None,r_forced = None):
        # ORDER PLANES
        #------------------------
        Thetax = 2*np.pi*Qx*N
        Thetay = 2*np.pi*Qy*N

        main = [self.projection(plane,Thetax[[i]],Thetay[[i]],r_aspect=r_aspect,r_forced=r_forced)[0][0,0] for i in range(len(N))]
        sec  = [self.projection(plane,Thetax[[i]],Thetay[[i]],r_aspect=r_aspect,r_forced=r_forced)[1][0,0] for i in range(len(N))]
        #------------------------

        return np.array(main),np.array(sec)

    def slice(self,plane,Thetax,Thetay):
        # ORDER PLANES
        #------------------------
        theta_vec   = Thetax if plane == 'x' else Thetay
        theta_slice = Thetay if plane == 'x' else Thetax
        vec_plane   = 0 if plane == 'x' else 1
        slice_plane = 1 if plane == 'x' else 0
        if isinstance(theta_slice, float):
            theta_slice = [theta_slice]
        #------------------------

        sbys_x = []
        sbys_y = []
        for t0 in theta_slice:
            hx = sum([A * np.exp(1j * (n[vec_plane]*theta_vec + n[slice_plane]*t0))  for A,n in zip(self.Ax,self.nx)])
            hy = sum([A * np.exp(1j * (n[vec_plane]*theta_vec + n[slice_plane]*t0))  for A,n in zip(self.Ay,self.ny)])
         
            sbys_x.append([np.real(hx),-np.imag(hx)])
            sbys_y.append([np.real(hy),-np.imag(hy)])

        return np.array(sbys_x).transpose(0, 2, 1),np.array(sbys_y).transpose(0, 2, 1)


    def projection(self,plane,Thetax,Thetay,r_aspect = None,r_forced = None,scale_transverse=1):
        # ORDER PLANES
        #------------------------
        theta_vec   = Thetax if plane == 'x' else Thetay
        theta_slice = Thetay if plane == 'x' else Thetax
        vec_plane   = 0 if plane == 'x' else 1
        slice_plane = 1 if plane == 'x' else 0

        if r_forced is not None:
            CoT = r_forced
        elif r_aspect is not None:
            other_I = self.Ix if plane == 'x' else self.Iy
            CoT = r_aspect*other_I
        else:
            CoT = self.Iy if plane == 'x' else self.Ix
            
        if isinstance(theta_slice, float):
            theta_slice = [theta_slice]
        #------------------------
            
        sbys_x,sbys_y = self.slice(plane,Thetax,Thetay)
        sbys_main = sbys_x if plane == 'x' else sbys_y
        sbys_sec  = sbys_y if plane == 'x' else sbys_x

        main_projection = []
        sec_projection  = []

        for s,t0 in enumerate(theta_slice):    
            center = np.array([[CoT*np.cos(t0)],[CoT*np.sin(t0)],[0]])

            proj_main = center + scale_transverse*np.array([sbys_main[s][:,0]*np.cos(t0),
                                                            sbys_main[s][:,0]*np.sin(t0),
                                                            sbys_main[s][:,1]])
            proj_sec  = center + scale_transverse*np.array([sbys_sec[s][:,0]*np.cos(t0),
                                                            sbys_sec[s][:,0]*np.sin(t0),
                                                            sbys_sec[s][:,1]])
            main_projection.append(proj_main)
            sec_projection.append(proj_sec)

        return np.array(main_projection).transpose(0, 2, 1),np.array(sec_projection).transpose(0, 2, 1),CoT
    

    def to_mesh(self,plane,num_angles,num_slices,r_aspect = None,r_forced = None,scale_transverse = 1):

        

        theta_vec   = np.linspace(0,2*np.pi,num_angles)
        theta_slice = np.linspace(0,2*np.pi,num_slices)
        if plane == 'x':
            main,sec,CoT = self.projection('x',theta_vec,theta_slice,r_aspect,r_forced)
            main_in,sec_in,_  = self.projection('x',theta_vec,theta_slice,r_aspect,r_forced,scale_transverse = scale_transverse)
        elif plane == 'y':
            main,sec,CoT = self.projection('y',theta_slice,theta_vec,r_aspect,r_forced)
            main_in,sec_in,_  = self.projection('y',theta_slice,theta_vec,r_aspect,r_forced,scale_transverse = scale_transverse)

        
        # OUTER SURFACE
        _xyz    = main
        v_idx_out   = np.arange(_xyz.shape[0]*_xyz.shape[1]).reshape((_xyz.shape[0],_xyz.shape[1]))
        
        verts_out   = _xyz.reshape(-1, _xyz.shape[-1]).tolist()
        faces_out   =[[ v_idx_out[s  ,i  ],
                    v_idx_out[s  ,i+1],
                    v_idx_out[s+1,i+1],
                    v_idx_out[s+1,i  ]]  for s in range(-1,num_slices-1) for i in range(-1,num_angles-1)]
        
        # INNER SURFACE
        _xyz    = main_in
        v_idx_in= np.arange(_xyz.shape[0]*_xyz.shape[1]).reshape((_xyz.shape[0],_xyz.shape[1]))
        
        verts_in= _xyz.reshape(-1, _xyz.shape[-1]).tolist()
        faces_in=[[ v_idx_in[s  ,i  ],
                    v_idx_in[s+1,i  ],
                    v_idx_in[s+1,i+1],
                    v_idx_in[s  ,i+1],
                    ]  for s in range(-1,num_slices-1) for i in range(-1,num_angles-1)]

        #===================
        #   s,i+1       s+1,i+1  
        #    +---------+  
        #    |         |  
        #    |    F    |  
        #    |         |  
        #    +---------+  
        #   s,i        s+1,i  
        #===================

        _mesh = Mesh(verts_in,faces_in,verts_out,faces_out)
        _mesh.meta['r'] = CoT


        _xyz    = sec
        v_idx_out   = np.arange(_xyz.shape[0]*_xyz.shape[1]).reshape((_xyz.shape[0],_xyz.shape[1]))
        
        verts_out   = _xyz.reshape(-1, _xyz.shape[-1]).tolist()
        faces_out   =[[ v_idx_out[s  ,i  ],
                    v_idx_out[s  ,i+1],
                    v_idx_out[s+1,i+1],
                    v_idx_out[s+1,i  ]]  for s in range(-1,num_slices-1) for i in range(-1,num_angles-1)]
        
        # INNER SURFACE
        _xyz    = sec_in
        v_idx_in= np.arange(_xyz.shape[0]*_xyz.shape[1]).reshape((_xyz.shape[0],_xyz.shape[1]))
        
        verts_in= _xyz.reshape(-1, _xyz.shape[-1]).tolist()
        faces_in=[[ v_idx_in[s  ,i  ],
                    v_idx_in[s+1,i  ],
                    v_idx_in[s+1,i+1],
                    v_idx_in[s  ,i+1],
                    ]  for s in range(-1,num_slices-1) for i in range(-1,num_angles-1)]
        
        _mesh_sec = Mesh(verts_in,faces_in,verts_in,faces_in)
        _mesh_sec.meta['r'] = CoT
        return {'main':_mesh,'sec':_mesh_sec}




class Mesh():
    def __init__(self,verts_in,faces_in,verts_out,faces_out):
        self.verts_in = verts_in
        self.faces_in = faces_in
        self.edges_in = []
        self.verts_out = verts_out
        self.faces_out = faces_out
        self.edges_out = []
        self.meta = {}

    # def to_dict(self):
    #     metadata = {'verts':self.verts,'faces':self.faces, 'edges':self.edges,'meta':self.meta}
    #     return metadata
    def to_dict(self):
        # Directly return a copy of the object's __dict__ attribute
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