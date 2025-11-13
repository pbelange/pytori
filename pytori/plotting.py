import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ============================================================
# Torus loop visualization
# ============================================================
def loop(torus, proj='x', looping=None, num_points=None, Tx=None, Ty=None, Tz=None, ax=None, add_artist=True, **kwargs):
    """
    Plot a phase-space loop (X, PX) from a Torus object using Matplotlib.

    Parameters
    ----------
    torus : pytori.Torus
        The Torus instance to visualize.
    proj : {'x', 'y', 'z'}
        Which phase-space projection to plot.
    looping : {'x', 'y', 'z'}, optional
        Which angle variable to sweep (default: 'x').
    num_points : int, optional
        Number of points along the loop (default: 200).
    Tx, Ty, Tz : float or array, optional
        Fixed angular coordinates of the other planes.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. Created if None.
    **kwargs :
        Passed to `Polygon` (facecolor, edgecolor, lw, alpha, ...).

    Returns
    -------
    patch : matplotlib.patches.Polygon
        The polygonal loop added to the axis.
    """

    assert proj in {'x', 'y', 'z'}, "proj must be one of {'x','y','z'}"
    
    # --- Get the angles ---
    if any(T is not None for T in (Tx, Ty, Tz)):
        # explicit angles
        assert looping is None, "If Tx, Ty, or Tz is given, looping must be None."
        assert num_points is None, "If Tx, Ty, or Tz is given, num_points must be None."
        Tx = 0 if Tx is None else Tx
        Ty = 0 if Ty is None else Ty
        Tz = 0 if Tz is None else Tz
    else:
        # generated loop
        looping = looping or 'x'
        num_points = num_points or 200
        Ts = {'x': 0, 'y': 0, 'z': 0}
        Ts[looping] = np.linspace(0, 2 * np.pi, num_points)
        Tx, Ty, Tz = Ts['x'], Ts['y'], Ts['z']


    bet = getattr(torus, f'bet{proj}0')
    ev  = torus.eval(plane=proj, Tx=Tx, Ty=Ty, Tz=Tz)
    Q,P = np.real(ev)*np.sqrt(bet), -np.imag(ev)/np.sqrt(bet)

    # --- get axis if needed ---
    if ax is None:
        ax = plt.gca()

    # --- Default visual style ---
    kwargs.setdefault('facecolor', 'none')
    kwargs.setdefault('edgecolor', 'C0')
    kwargs.setdefault('lw', 1.5)
    

    # --- Create polygon patch ---
    coords  = np.column_stack((Q, P))
    patch   = Polygon(coords, closed=True, **kwargs)
    
    # --- Add to axes ---
    if add_artist:
        ax.add_patch(patch)
        ax.relim()
        ax.autoscale_view()


    return patch


def xloop(torus, **kwargs):
    """Shortcut to plot x-Px loop from Torus."""
    return loop(torus, proj='x', **kwargs)
def yloop(torus, **kwargs):
    """Shortcut to plot y-Py loop from Torus."""
    return loop(torus, proj='y', **kwargs)
def zloop(torus, **kwargs):
    """Shortcut to plot z-Pz loop from Torus."""
    return loop(torus, proj='z', **kwargs)
# ============================================================



# Mesh



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
        



        # Preparing argument dictionnary
        angle_dict = {'Tx':Tx, 'Ty':Ty, 'Tz':Tz}
        angle_dict.update({f'T{plane_ij[0]}': theta_angle})

        if r is None:
            r = np.sqrt(2*torus.Ij(plane_ij[0]))
        r *= r_rescale

        # Generating the slices
        slices = []
        slices_in = []
        slice_offset = angle_dict[f'T{slice_along}']
        for t0 in theta_slice:
            # Coordinates
            angle_dict.update({f'T{slice_along}': t0+slice_offset})
            evals = torus.eval(plane_ij[1],**angle_dict) 
            x,z = np.real(evals)*np.sqrt(torus.betx0), -np.imag(evals)/np.sqrt(torus.betx0)
            
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



# Generic functions

###############################################################################
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import FancyArrowPatch
def drawArrow(x,y,scale=2,rotate=0,facecolor = None,color='C0',alpha=1,label = None,zorder=None,arrow_dict = None,**kwargs):
    ax = plt.gca()
    ax.plot(x[:-2], y[:-2], color=color,alpha=alpha,label=label,**kwargs)
    posA, posB = zip(x[-2:], y[-2:])
    edge_width = 2.*scale
    anglestyle = "arc3,rad={}".format(np.radians(rotate))
    #arrowstyle was 3*edge_width,3*edge_width,edge_width before.
    head_length,head_width,tail_width = 3*edge_width, 2*edge_width, 2*edge_width
    if arrow_dict is not None:
        head_length,head_width,tail_width = arrow_dict['head_length'],arrow_dict['head_width'],arrow_dict['tail_width']
    arrowstyle = "fancy,head_length={},head_width={},tail_width={}".format(head_length,head_width,tail_width)

    if facecolor is None:
        arrow = FancyArrowPatch(posA=posA, posB=posB, arrowstyle=arrowstyle, connectionstyle=anglestyle,color=color,alpha=alpha,zorder=zorder)
    else:
        arrow = FancyArrowPatch(posA=posA, posB=posB, arrowstyle=arrowstyle, connectionstyle=anglestyle,facecolor=facecolor,edgecolor=color,alpha=alpha,zorder=zorder,linewidth=0.3)
    ax.add_artist(arrow)
#################################################################################


###############################################################################
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import FancyArrowPatch
def drawVector(vec,O = [0,0],scale=2,rotate=0,facecolor = None,color='C0',alpha=1,label = None,zorder=None,arrow_dict = None):
    ax = plt.gca()

    if isinstance(vec,(complex,np.complex128)):
        vec = [np.real(vec),-np.imag(vec)]
    if isinstance(O,(complex,np.complex128)):
        O = [np.real(O),-np.imag(O)]

    x_vec = [O[0],1*(vec[0]+O[0])]
    y_vec = [O[1],1*(vec[1]+O[1])]
    posA, posB = zip(x_vec,y_vec)

    

    # Arrow format
    #----------------------------------
    edge_width = 2.*scale
    anglestyle = "arc3,rad={}".format(np.radians(rotate))
    head_length,head_width,tail_width = 3*edge_width, 2*edge_width, 0.1*edge_width
    if arrow_dict is not None:
        head_length,head_width,tail_width = arrow_dict['head_length'],arrow_dict['head_width'],arrow_dict['tail_width']
    arrowstyle = "fancy,head_length={},head_width={},tail_width={}".format(head_length,head_width,tail_width)
    #----------------------------------


    shrink = 1
    ax.plot(x_vec,y_vec, color=color,alpha=0,label=label)
    if facecolor is None:
        arrow = FancyArrowPatch(posA=posA, posB=posB,shrinkB=shrink,shrinkA=shrink, arrowstyle=arrowstyle, connectionstyle=anglestyle,color=color,alpha=alpha,zorder=zorder)
    else:
        arrow = FancyArrowPatch(posA=posA, posB=posB,shrinkB=shrink,shrinkA=shrink, arrowstyle=arrowstyle, connectionstyle=anglestyle,facecolor=facecolor,edgecolor=color,alpha=alpha,zorder=zorder)

    ax.add_artist(arrow)
#==============================================================================
    




from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs, ys, zs = self._verts3d
        xs2d, ys2d, _ = proj3d.proj_transform(xs, ys, zs, self.axes.M)
        self.set_positions((xs2d[0], ys2d[0]), (xs2d[1], ys2d[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        # Return depth value used for z-ordering
        xs, ys, zs = self._verts3d
        _, _, z = proj3d.proj_transform(xs, ys, zs, self.axes.M)
        return np.mean(z)  # Or min(z), max(z), etc.



def drawArrow3D(x, y, z, scale=2, color='C0', alpha=1, zorder=None, arrow_dict=None, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    ax.plot(x[:-1], y[:-1], z[:-1], color=color, alpha=alpha, **kwargs)

    # Define arrowhead style
    edge_width = 2. * scale
    head_length = 3 * edge_width
    head_width = 2 * edge_width
    tail_width = 2 * edge_width
    if arrow_dict is not None:
        head_length = arrow_dict.get('head_length', head_length)
        head_width = arrow_dict.get('head_width', head_width)
        tail_width = arrow_dict.get('tail_width', tail_width)

    arrowstyle = f"fancy,head_length={head_length},head_width={head_width},tail_width={tail_width}"

    # Get arrow tail and head
    xs, ys, zs = x[-2:], y[-2:], z[-2:]

    arrow = Arrow3D(xs, ys, zs,
                    mutation_scale=10,
                    arrowstyle=arrowstyle,
                    color=color,
                    alpha=alpha,
                    zorder=zorder)
    ax.add_artist(arrow)



def add_plane_canvas(z,xlim,ylim,facecolors='whitesmoke', edgecolors='gray',ax=None,ds=None):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if ax is None:
        ax = plt.gca()

    if ds is None:
        ds = 1e-2 * np.abs(np.diff(ax.get_zlim()))[0]/10


    z = z - ds
    x0, x1 = xlim
    y0, y1 = ylim
    
    

    plane_rect = [
        [x0, y0, z],
        [x1, y0, z],
        [x1, y1, z],
        [x0, y1, z]
    ]

    # Creating patch
    plane_patch = Poly3DCollection([plane_rect], facecolors=facecolors, edgecolors=edgecolors,zsort='min')#,shade=False)
    plane_patch.set_zorder(100+z)
    plane_patch.set_label(f"plane_at_z={z}")
    ax.add_collection3d(plane_patch)



def add_oriented_plane(center, normal, width, height, ax=None,
                       facecolor='whitesmoke', edgecolor='k',linewidths=1, alpha=1.0,zorder=100,f_alpha=None,e_alpha=None):
    """
    Add a rectangular plane in 3D space centered at `center`, with normal `normal`,
    and dimensions `width` (along u) and `height` (along v).

    Parameters
    ----------
    center : (3,) array-like
        The (x, y, z) coordinates of the center of the plane.
    normal : (3,) array-like
        The normal vector to the plane (does not need to be normalized).
    width : float
        Width of the plane (along u direction).
    height : float
        Height of the plane (along v direction).
    ax : Axes3D
        Matplotlib 3D axis.
    facecolor, edgecolor : str or tuple
        Matplotlib colors.
    alpha : float
        Transparency.
    """
    if ax is None:
        ax = plt.gca()

    center = np.asarray(center)
    normal = np.asarray(normal)
    normal = normal / np.linalg.norm(normal)

    # Create two orthogonal vectors in the plane
    if np.allclose(normal, [0, 0, 1]):
        u = np.array([1, 0, 0])
    else:
        u = np.cross(normal, [0, 0, 1])
        u = u / np.linalg.norm(u)
    v = np.cross(normal, u)

    # print(type(u),type(v))
    u = u * width / 2
    v = v * height / 2

    # Define corners of the rectangle
    corners = [
        center - u - v,
        center + u - v,
        center + u + v,
        center - u + v,
    ]

    patch = Poly3DCollection([corners], facecolor=facecolor, edgecolor=edgecolor, alpha=alpha,linewidths=linewidths)
    patch.set_zorder(zorder)
    ax.add_collection3d(patch)



def plot_proj(x,y,z,ax=None,**kwargs):
    if ax is None:
        ax = plt.gca()


    if 'zorder' not in kwargs.keys():
        kwargs['zorder'] = 100 + z
    plt.plot(x, y, zs=z, zdir='z',**kwargs)


def plot_normal_proj(x, y, center, normal, ax=None,flip_y = False, **kwargs):
    """
    Plots a 2D (x, y) curve on an oriented 3D plane defined by a center and normal vector.

    Parameters
    ----------
    x, y : array-like
        2D coordinates in the plane's local coordinate system.
    center : array-like
        3D center of the target plane.
    normal : array-like
        Normal vector defining the plane's orientation.
    ax : Axes3D
        The matplotlib 3D axis to plot on. If None, uses current axis.
    kwargs : dict
        Passed directly to `ax.plot()`.
    """
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D

    if ax is None:
        ax = plt.gca()

    # # Normalize the normal vector
    # normal = np.asarray(normal, dtype=float)
    # normal /= np.linalg.norm(normal)

    # # Create orthonormal basis (u, v) in the plane
    # u = np.cross(normal, [0, 0, 1])
    # if np.linalg.norm(u) < 1e-8:
    #     u = np.array([1, 0, 0])
    # u /= np.linalg.norm(u)
    # v = -np.cross(normal, u)

    # if flip_y:
    #     v = -v

    center = np.asarray(center)
    normal = np.asarray(normal)
    normal = normal / np.linalg.norm(normal)

    # Create two orthogonal vectors in the plane
    if np.allclose(normal, [0, 0, 1]):
        u = np.array([1, 0, 0])
    else:
        u = np.cross(normal, [0, 0, 1])
        u = u / np.linalg.norm(u)
    v = -np.cross(normal, u)

    if flip_y:
        v = -v


    # Project 2D (x, y) onto the 3D plane
    pts3d = np.outer(x, u) + np.outer(y, v) + center

    ax.plot(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2], **kwargs)


def plot_poly_proj(poly2d,center, normal,ax=None,zorder=None,flip_y = False,export=False,**kwargs):
    """
    Add a rectangular plane in 3D space centered at `center`, with normal `normal`,
    and dimensions `width` (along u) and `height` (along v).

    Parameters
    ----------
    center : (3,) array-like
        The (x, y, z) coordinates of the center of the plane.
    normal : (3,) array-like
        The normal vector to the plane (does not need to be normalized).
    width : float
        Width of the plane (along u direction).
    height : float
        Height of the plane (along v direction).
    ax : Axes3D
        Matplotlib 3D axis.
    facecolor, edgecolor : str or tuple
        Matplotlib colors.
    alpha : float
        Transparency.
    """
    if ax is None:
        ax = plt.gca()

    center = np.asarray(center)
    normal = np.asarray(normal)
    normal = normal / np.linalg.norm(normal)

    # Create two orthogonal vectors in the plane
    if np.allclose(normal, [0, 0, 1]):
        u = np.array([1, 0, 0])
    else:
        u = np.cross(normal, [0, 0, 1])
        u = u / np.linalg.norm(u)
    v = -np.cross(normal, u)

    if flip_y:
        v = -v



    # Get poly3d coordinates
    poly3d = []
    for vert in poly2d.get_path().vertices:
        poly3d.append(center + vert[0]*u + vert[1]*v)


    if 'facecolors' in kwargs:
        if kwargs['facecolors'] == 'none':
            kwargs['facecolors'] = (0, 0, 0, 0)
    if 'facecolor' in kwargs:
        if kwargs['facecolor'] == 'none':
            kwargs['facecolor'] = (0, 0, 0, 0)

    patch = Poly3DCollection([poly3d], **kwargs)
    if zorder is not None:
        patch.set_zorder(zorder)
    ax.add_collection3d(patch)

    if export:
        return patch,poly3d


def compute_face_normals(poly3d):
    normals = []
    for face in poly3d:
        if len(face) < 3:
            normals.append(np.array([0, 0, 1]))  # fallback
            continue
        v0, v1, v2 = np.asarray(face[0]), np.asarray(face[1]), np.asarray(face[2])
        n = np.cross(v1 - v0, v2 - v0)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            normals.append(np.array([0, 0, 1]))  # fallback direction
        else:
            normals.append(n / norm)
    return np.array(normals)


def add_light_arrows(lights, ax=None, length=None, color='gold'):
    """
    Draw 3D arrows from light source (in direction used by LightSource) to the origin.

    Parameters
    ----------
    lights : list of (LightSource, weight)
        Light sources to visualize.
    ax : Axes3D
        The 3D matplotlib axis to draw on.
    length : float or None
        Length of each light vector. If None, automatically estimated from plot limits.
    color : str
        Color of the light arrows.
    """
    import numpy as np

    if ax is None:
        ax = plt.gca()

    if length is None:
        xlim, ylim, zlim = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
        diag = np.linalg.norm([
            xlim[1] - xlim[0],
            ylim[1] - ylim[0],
            zlim[1] - zlim[0]
        ])
        length = diag/2


    

    for ls, weight in lights:
        # Basis vectors
        nx = np.array([[1, 0, 0]])
        ny = np.array([[0, 1, 0]])
        nz = np.array([[0, 0, 1]])

        # Light intensities along each axis
        lx = ls.shade_normals(nx)[0]  # ~dot(-light, x)
        ly = ls.shade_normals(ny)[0]
        lz = ls.shade_normals(nz)[0]

        # Combine into vector and normalize
        lvec = np.array([lx, ly, lz])
        norm = np.linalg.norm(lvec)
        

        light_dir = lvec / norm if norm > 0 else lvec

        start = light_dir * length
        end   = np.zeros(3)
        delta = end - start

        ax.quiver(
            start[0], start[1], start[2],
            delta[0], delta[1], delta[2],
            color=color,
            linewidth=10*weight,  # Scale arrow width by weight
            arrow_length_ratio=0.1
        )

        # xs, ys, zs = zip(start, end)
        # ax.plot(xs, ys, zs, color=color, linewidth=linewidth)

from matplotlib.colors import LightSource
def apply_multilight_shading(collection, poly3d, lights=None,specular=1,contrast=1,preset=None):
    """
    Apply multi-light shading to a Poly3DCollection while preserving its original facecolor and alpha.

    Parameters
    ----------
    collection : Poly3DCollection
        The mesh to which lighting will be applied.
    poly3d : list of faces (each a list of 3D vertices)
        The geometry for computing face normals.
    lights : list of (LightSource, weight)
        Lighting directions and their respective intensities.
    """


    style_params = {
        "metallic": {
            "lights": [ (LightSource(azdeg=45, altdeg=60), 0.9),
                        (LightSource(azdeg=135, altdeg=20), 0.1)],
            "contrast": 1.2,
            "specular": 1.5,
        },
        "soft": {
            "lights": [
                (LightSource(azdeg=45, altdeg=60), 0.8),
                (LightSource(azdeg=135, altdeg=20), 0.2),
            ],
            "contrast": 0.9,
            "specular": 0.9,
        },
        "flat": {
            "lights": [(LightSource(azdeg=45, altdeg=60), 1.0)],
            "contrast": 1.0,
            "specular": 1.0,
        }
    }

    if preset is not None:
        if preset not in style_params:
            raise ValueError(f"Unknown preset '{preset}'. Available presets: {list(style_params.keys())}")
        lights = style_params[preset]["lights"]
        contrast = style_params[preset]["contrast"]
        specular = style_params[preset]["specular"]


    # Compute normals
    normals = compute_face_normals(poly3d)
    N = len(normals)

    # Extract the base color (assuming uniform facecolor)
    original_fc = collection.get_facecolor()
    if original_fc.shape[0] == 0:
        raise ValueError("Collection has no facecolor set.")
    base_rgba = original_fc[0]  # shape (4,)
    base_rgb, alpha = base_rgba[:3], base_rgba[3]

    # Initialize shaded RGB array
    shaded = np.zeros((N, 3))

    for ls, weight in lights:
        intensity = ls.shade_normals(normals, fraction=contrast)  # shape (N,)
        if specular != 1:
            intensity = np.clip(intensity ** specular, 0, 1) 
        shaded += weight * intensity[:, np.newaxis]  # broadcast to (N,3)

    # Multiply base color by combined light intensity
    final_rgb = np.clip(shaded * base_rgb[None, :], 0, 1)
    final_rgba = np.concatenate([final_rgb, np.full((N, 1), alpha)], axis=1)

    
    collection.set_facecolors(final_rgba)
    collection.set_edgecolors(final_rgba)

    return lights,final_rgba




def merge_collections(coll1,coll2,mesh1,mesh2,color1,color2,ax=None,zorder=None, **kwargs):
    """
    Merge two Poly3DCollection objects into one, preserving their face colors.
    """
    # Pick an axes if not provided
    if ax is None:
        ax = plt.gca()


    line_width      = coll1.get_linewidths()  # Returns array of line widths
    alpha_val       = coll1.get_alpha() 
    is_rasterized   = coll1.get_rasterized()   # True / False
    is_aliased      = coll1.get_antialiaseds()
    old_zorder      = coll1.get_zorder()  # Get the zorder of the first collection

    # Remove originals safely
    coll1.remove()
    coll2.remove()

    # all_poly3d = poly3d_1 + poly3d_2
    mesh = merge_meshes(mesh1, mesh2)
    all_poly3d = mesh.poly3d
    all_colors = np.vstack([color1, color2])

    merged = Poly3DCollection(all_poly3d, facecolors=all_colors, **kwargs)
    if zorder is not None:
        merged.set_zorder(zorder)
    else:
        merged.set_zorder(old_zorder)

    # reset properties
    merged.set_edgecolors(all_colors)
    merged.set_linewidths(line_width)
    # Apply alpha only if colors don't already have alpha
    if all_colors.shape[1] == 3 and alpha_val is not None:
        merged.set_alpha(alpha_val)
    merged.set_rasterized(is_rasterized)
    merged.set_antialiaseds(is_aliased)


    
    # ax.add_collection3d(merged)
    return merged,mesh,all_colors




def merge_meshes(m1, m2):
    import pytori as pt
    """
    Merge two Mesh objects into a new Mesh.
    The new mesh will have the polycenter of m1.
    """

    # Offset for face indices of the second mesh
    offset = len(m1.verts)
    verts = m1.verts + m2.verts
    faces = m1.faces + [[idx + offset for idx in face] for face in m2.faces]

    # Inner surface (if both meshes have them)
    verts_in = None
    faces_in = None
    if m1.verts_in is not None and m2.verts_in is not None:
        offset_in = len(m1.verts_in)
        verts_in = m1.verts_in + m2.verts_in
        faces_in = m1.faces_in + [[idx + offset_in for idx in face] for face in m2.faces_in]

    # Merge meta data and radius
    r = max(m1.r, m2.r)
    polycenter = m1.polycenter.copy()  # Keep the polycenter of the first mesh

    return Mesh(r, verts, faces, verts_in=verts_in, faces_in=faces_in, polycenter=polycenter)



def add_ribbon(radius=1.0, theta = None, center = [0,0,0], height=0, width=0.1, turns=1, color='k', alpha=1.0, n=200,ax=None,zorder=None,arrow_rescale=1):
    """
    Add a 3D ribbon wrapping around the z-axis, with its normal
    always perpendicular to the axis.
    """

    if ax is None:
        ax = plt.gca()

    if theta is None:
        theta   = np.linspace(0, 2 * np.pi * turns, n)
        z       = np.linspace(0, height, n)
    else:
        # theta = np.asarray(theta)
        n = len(theta)
        z = np.linspace(0, height, n)
        turns = np.max([len(theta) / (2 * np.pi),1])

    centerline = np.column_stack((radius * np.cos(theta) + center[0],
                                  radius * np.sin(theta) + center[1],
                                  z + center[2]))

    faces = []
    for i in range(n - 1):
        p0 = centerline[i]
        p1 = centerline[i + 1]

        # Tangent
        t = p1 - p0
        t /= np.linalg.norm(t)

        # Radial direction (normal to axis)
        nvec = np.array([p0[0], p0[1], 0])
        nvec /= np.linalg.norm(nvec)

        # Binormal for ribbon width
        b = np.cross(t, nvec)
        b /= np.linalg.norm(b)

        # Offset left/right
        p0_left = p0 + (width / 2) * b
        p0_right = p0 - (width / 2) * b
        p1_left = p1 + (width / 2) * b
        p1_right = p1 - (width / 2) * b

        faces.append([p0_left, p1_left, p1_right, p0_right])

    ribbon = Poly3DCollection(faces, facecolor=color, alpha=alpha, edgecolor=color,linewidths=radius*np.min(np.diff(theta))/5)
    if zorder is not None:
        ribbon.set_zorder(zorder)
    else:
        zorder = 1



    if arrow_rescale != 0:
        t_arr = np.linspace(theta[-10], theta[-1] + 1.5*np.min(np.diff(theta)), 10)
        z_arr = np.linspace(z[-10], z[-1] + 1.5*np.min(np.diff(z)), 10)
        c_arr = np.column_stack((   radius * np.cos(t_arr) + center[0],
                                    radius * np.sin(t_arr) + center[1],
                                    z_arr + center[2]))
        size = width*arrow_rescale
        drawArrow3D(c_arr[-5:,0],c_arr[-5:,1],c_arr[-5:,2],color=color,lw=0,alpha=alpha,zorder=zorder,arrow_dict={'head_length':0.8*size,'head_width':0.8*size,'tail_width':0.8*size})


    # ribbon.set_rasterized(True)  # Rasterize for performance
    # ribbon.set_antialiaseds(False)  # Smooth edges
    # ribbon.set_edgecolor((0,0,0,0))
    # ribbon.set_edgecolor(color)
    # ribbon.set_linewidths(0.1) 
    ax.add_collection3d(ribbon)
    return ribbon



def rescale_loop(loop,x_mult,y_mult):
    from matplotlib.transforms import Affine2D

    # Get the original path vertices (in *local* coordinates)
    verts = loop.get_path().vertices.copy()

    # Apply scaling directly
    verts_scaled = verts * [x_mult,y_mult]

    # Create a new path with the scaled vertices
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch

    new_path = Path(verts_scaled, loop.get_path().codes)
    loop.set_path(new_path)