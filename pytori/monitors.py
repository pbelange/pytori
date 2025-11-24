import numpy as np
import pandas as pd
from pathlib import Path
import nafflib

import xtrack as xt
import xpart as xp
import xobjects as xo





# Adding reset method to monitors:
#==============================
def reset_monitor(self,start_at_turn = None,stop_at_turn = None):
    if start_at_turn is not None:
        self.start_at_turn = start_at_turn
    if stop_at_turn is not None:
        self.stop_at_turn = stop_at_turn
    
    with self.data._bypass_linked_vars():
            for tt, nn in self._ParticlesClass.per_particle_vars:
                getattr(self.data, nn)[:] = 0

if not hasattr(xt.ParticlesMonitor, "reset"):
    xt.ParticlesMonitor.reset = reset_monitor
else:
    # Method exists — check if it is OUR patch
    if xt.ParticlesMonitor.reset is not reset_monitor:
        raise RuntimeError(
            "`reset` method already exists and is not from pytori."
        )
#==============================


#==============================
def phys2norm(self,twiss_init,nemitt_x=None,nemitt_y=None,nemitt_zeta=None):

        table = twiss_init.get_normalized_coordinates(self,nemitt_x=nemitt_x, nemitt_y=nemitt_y,nemitt_zeta=nemitt_zeta)

        monitor_norm = self.copy()
        norm_vars = tuple((tt,nn+'_norm') for tt,nn in xt.Particles.per_particle_vars+((xo.Float64, 'pzeta'),) if nn in ['x','y','zeta','px','py','pzeta'])
        for tt, nn in norm_vars:
            setattr(monitor_norm, nn, xt.monitors.particles_monitor._FieldOfMonitor(name=nn))

        monitor_norm.x_norm = table.x_norm.reshape(self.x.shape)
        monitor_norm.y_norm = table.y_norm.reshape(self.y.shape)
        monitor_norm.zeta_norm = table.zeta_norm.reshape(self.zeta.shape)
        monitor_norm.px_norm = table.px_norm.reshape(self.px.shape)
        monitor_norm.py_norm = table.py_norm.reshape(self.py.shape)
        monitor_norm.pzeta_norm = table.pzeta_norm.reshape(self.pzeta.shape)
        
        return monitor_norm

xt.ParticlesMonitor.phys2norm = phys2norm

if not hasattr(xt.ParticlesMonitor, "phys2norm"):
    xt.ParticlesMonitor.phys2norm = phys2norm
else:
    # Method exists — check if it is OUR patch
    if xt.ParticlesMonitor.phys2norm is not phys2norm:
        raise RuntimeError(
            "`phys2norm` method already exists and is not from pytori."
        )
#==============================



#===================================================
# BASE CLASS
#===================================================
class Buffer():
    def __init__(self,):
        self.call_ID = None
        # Data dict to store whatever data
        self.data = {}
        # Particle ID to keep track
        self.particle_id = None
        self.complex2tuple = True


    def to_dict(self):
        dct    = {}
        nparts = len(self.particle_id)

        for key,value in self.data.items():

            if len(np.shape(value)) == 1:
                dct[key] = np.repeat(value,nparts)
            elif len(np.shape(value)) == 2:
                dct[key] = np.hstack(value)
            elif len(np.shape(value)) == 3:
                # numpy array for each particle
                if np.issubdtype(value[0].dtype, complex) and self.complex2tuple:
                    # is complex
                    dct[key] = [[(c.real, c.imag) for c in row] for row in np.vstack(value).tolist()]
                else:
                    dct[key] = np.vstack(value).tolist()
            else:
                pass

        return dct
    
    def to_pandas(self):
        return pd.DataFrame(self.to_dict())
    

    def update(self,monitor):
        # Initialize
        #-------------------------
        if self.call_ID is None:
            self.call_ID = 0
        else:
            self.call_ID += 1
        
        if self.particle_id is None:
            self.particle_id = np.arange(monitor.part_id_start,monitor.part_id_end)
        #-------------------------
#===================================================



#===================================================
# To apply NAFF on a monitor directly
#===================================================
class NAFFMonitor(Buffer):
    def __init__(self,normalize=True,complex2tuple=False):
        super().__init__()  
        self.clean()
        self.normalize = normalize
        self.complex2tuple = complex2tuple

        # To be injected manually!
        #=========================
        self.twiss          = None
        self.nemitt_x       = None
        self.nemitt_y       = None
        self.nemitt_zeta    = None
        #=========================

        # NAFF parameters
        #=========================
        self.n_harm       = None
        self.window_order = None
        self.window_type  = None
        self.multiprocesses = None
        self.normalize = normalize
        #=========================


    def clean(self,):
        self.data['window'] = []
        self.data['particle'] = []
        self.data['start_at_turn'] = []
        self.data['stop_at_turn']  = []
        self.data['N'] = []
        self.data['state']    = []

        self.data['Ax']  = []
        self.data['Qx']  = []
        self.data['Ay']  = []
        self.data['Qy']  = []
        self.data['Azeta']  = []
        self.data['Qzeta']  = []

    def process(self,monitor):
        self.update(monitor = monitor)


        # Extracting data
        #-------------------------
        start_at_turn = monitor.start_at_turn
        stop_at_turn  = monitor.stop_at_turn


        if self.normalize:
            monitor = monitor.phys2norm(    twiss_init    = self.twiss,
                                            nemitt_x      = self.nemitt_x,
                                            nemitt_y      = self.nemitt_y,
                                            nemitt_zeta   = self.nemitt_zeta)
            x_sig       = monitor.x_norm
            px_sig      = monitor.px_norm
            y_sig       = monitor.y_norm
            py_sig      = monitor.py_norm
            zeta_sig    = monitor.zeta_norm
            pzeta_sig   = monitor.pzeta_norm
        else:
            x_sig       = monitor.x
            px_sig      = monitor.px
            y_sig       = monitor.y
            py_sig      = monitor.py
            zeta_sig    = monitor.zeta
            pzeta_sig   = monitor.pzeta


        # Extracting the harmonics
        #--------------------------
        n_harm       = self.n_harm
        window_order = self.window_order
        window_type  = self.window_type

        Ax,Qx       = nafflib.multiparticle_harmonics(x_sig,px_sig      , num_harmonics=n_harm, window_order=window_order, window_type=window_type, processes = self.multiprocesses)
        Ay,Qy       = nafflib.multiparticle_harmonics(y_sig,py_sig      , num_harmonics=n_harm, window_order=window_order, window_type=window_type, processes = self.multiprocesses)
        Azeta,Qzeta = nafflib.multiparticle_harmonics(zeta_sig,pzeta_sig, num_harmonics=n_harm, window_order=window_order, window_type=window_type, processes = self.multiprocesses)


        # Appending to data
        #-------------------------
        self.data['window'].append(self.call_ID)
        self.data['particle'].append(self.particle_id)
        self.data['start_at_turn'].append(start_at_turn)
        self.data['stop_at_turn'].append(stop_at_turn)
        self.data['N'].append(len(x_sig[0]))
        self.data['state'].append(monitor.state[:,-1].astype('int').copy())
        #----------
        self.data['Ax'].append(Ax)
        self.data['Qx'].append(Qx)
        self.data['Ay'].append(Ay)
        self.data['Qy'].append(Qy)
        self.data['Azeta'].append(Azeta)
        self.data['Qzeta'].append(Qzeta)
        #-------------------------
#===================================================



#===================================================
# TORUS BUFFER
#===================================================
class TorusMonitor(Buffer):
    def __init__(self,normalize=True,complex2tuple=False,skip_naff = False):
        super().__init__()  
        self.clean()
        self.normalize      = normalize
        self.complex2tuple  = complex2tuple
        self.skip_naff      = skip_naff

        # To be injected manually!
        #=========================
        self.twiss          = None
        self.nemitt_x       = None
        self.nemitt_y       = None
        self.nemitt_zeta    = None
        #=========================

        # NAFF parameters
        #=========================
        self.n_torus      = None
        self.n_points     = None
        #-------------------------
        self.n_harm       = None
        self.window_order = None
        self.window_type  = None
        self.multiprocesses = None
        #=========================

    def to_dict(self):
        dct    = {}
        for key,value in self.data.items():
            if len(value) == 0:
                continue
            if np.issubdtype(value[0].dtype, complex) and self.complex2tuple:
                # is complex
                dct[key] = [[(c.real, c.imag) for c in row] for row in value]
            else:
                dct[key] = value.tolist()
        return dct
        
    def clean(self,):
        self.data['turn']   = []
        self.data['torus']  = []
        self.data['state']  = []

        self.data['Ax']  = []
        self.data['Qx']  = []
        self.data['Ay']  = []
        self.data['Qy']  = []
        self.data['Azeta']  = []
        self.data['Qzeta']  = []

        self.data['Jx']     = []
        self.data['Jy']     = []
        self.data['Jzeta']  = []
        self.data['Jcov']   = []
        self.data['Jsmear'] = []
        self.data['fcov']   = []
        self.data['fsmear'] = []

    def process(self,monitor):
        self.update(monitor = monitor)

        assert self.call_ID <= 1, "TORUS_Buffer is not designed to store multiple chunks!"


        # Extracting data
        #-------------------------
        start_at_turn = monitor.start_at_turn
        stop_at_turn  = monitor.stop_at_turn
        self.n_turns  = stop_at_turn-start_at_turn


        if self.normalize:
            monitor = monitor.phys2norm( twiss_init    = self.twiss,
                                    nemitt_x      = self.nemitt_x,
                                    nemitt_y      = self.nemitt_y,
                                    nemitt_zeta   = self.nemitt_zeta)
            x_sig       = monitor.x_norm
            px_sig      = monitor.px_norm
            y_sig       = monitor.y_norm
            py_sig      = monitor.py_norm
            zeta_sig    = monitor.zeta_norm
            pzeta_sig   = monitor.pzeta_norm
        else:
            x_sig       = monitor.x
            px_sig      = monitor.px
            y_sig       = monitor.y
            py_sig      = monitor.py
            zeta_sig    = monitor.zeta
            pzeta_sig   = monitor.pzeta


        # Reshaping for faster handling
        #========================================
        torus_idx,turn_idx = np.mgrid[:self.n_torus,:self.n_turns]
        torus_idx = torus_idx.reshape(self.n_torus*self.n_turns)
        turn_idx  = turn_idx.reshape(self.n_torus*self.n_turns)
        state_multi = np.all(np.array(np.split(monitor.state.T, indices_or_sections = self.n_torus , axis=1)).reshape(self.n_torus*self.n_turns,self.n_points)==1,axis=1).astype(int)

        x_multi     = np.array(np.split(x_sig.T     , indices_or_sections = self.n_torus , axis=1)).reshape(self.n_torus*self.n_turns,self.n_points)
        px_multi    = np.array(np.split(px_sig.T    , indices_or_sections = self.n_torus , axis=1)).reshape(self.n_torus*self.n_turns,self.n_points)
        y_multi     = np.array(np.split(y_sig.T     , indices_or_sections = self.n_torus , axis=1)).reshape(self.n_torus*self.n_turns,self.n_points)
        py_multi    = np.array(np.split(py_sig.T    , indices_or_sections = self.n_torus , axis=1)).reshape(self.n_torus*self.n_turns,self.n_points)
        zeta_multi  = np.array(np.split(zeta_sig.T  , indices_or_sections = self.n_torus , axis=1)).reshape(self.n_torus*self.n_turns,self.n_points)
        pzeta_multi = np.array(np.split(pzeta_sig.T , indices_or_sections = self.n_torus , axis=1)).reshape(self.n_torus*self.n_turns,self.n_points)
        #========================================
        
        # Computing C-S like invariants
        Jx = 1/2 * np.mean(x_multi**2+px_multi**2,axis=1)
        Jy = 1/2 * np.mean(y_multi**2+py_multi**2,axis=1)
        Jzeta = 1/2 * np.mean(zeta_multi**2+pzeta_multi**2,axis=1)
        J_cov = 1/2 * np.mean((x_multi**2+px_multi**2)**2 + (y_multi**2+py_multi**2)**2 + (zeta_multi**2+pzeta_multi**2)**2,axis=1) - (Jx**2+Jy**2+Jzeta**2)
        J_smear = np.sqrt(J_cov)/np.sqrt(Jx**2+Jy**2+Jzeta**2)


        fx =  1/2 * (x_multi**2+px_multi**2 -  np.mean(x_multi**2+px_multi**2,axis=1)[:, np.newaxis])
        fy =  1/2 * (y_multi**2+py_multi**2 -  np.mean(y_multi**2+py_multi**2,axis=1)[:, np.newaxis])
        fzeta =  1/2 * (zeta_multi**2+pzeta_multi**2 -  np.mean(zeta_multi**2+pzeta_multi**2,axis=1)[:, np.newaxis])
        f_cov = np.mean(fx**2 + fy**2 + fzeta**2,axis=1)
        f_smear = np.sqrt(f_cov)/np.sqrt(Jx**2+Jy**2+Jzeta**2)

        if self.skip_naff or (self.n_harm is None) or (self.n_harm == 0):
            # Appending to data
            #-------------------------
            self.data['turn']   = turn_idx
            self.data['torus']  = torus_idx
            self.data['state']  = state_multi
            #----------
            self.data['Jx']     = Jx
            self.data['Jy']     = Jy
            self.data['Jzeta']  = Jzeta
            self.data['Jcov']   = J_cov
            self.data['Jsmear'] = J_smear
            self.data['fsmear'] = f_smear
            self.data['fcov']   = f_cov
            #-------------------------
        else:
            # Extracting the harmonics
            #--------------------------
            n_harm       = self.n_harm
            window_order = self.window_order
            window_type  = self.window_type

            Ax,Qx       = nafflib.multiparticle_harmonics(x_multi,px_multi      , num_harmonics=n_harm, window_order=window_order, window_type=window_type, processes = self.multiprocesses)
            Ay,Qy       = nafflib.multiparticle_harmonics(y_multi,py_multi      , num_harmonics=n_harm, window_order=window_order, window_type=window_type, processes = self.multiprocesses)
            Azeta,Qzeta = nafflib.multiparticle_harmonics(zeta_multi,pzeta_multi, num_harmonics=n_harm, window_order=window_order, window_type=window_type, processes = self.multiprocesses)



            # Appending to data
            #-------------------------
            self.data['turn']   = turn_idx
            self.data['torus']  = torus_idx
            self.data['state']  = state_multi
            #----------
            self.data['Ax'] = Ax
            self.data['Qx'] = Qx
            self.data['Ay'] = Ay
            self.data['Qy'] = Qy
            self.data['Azeta'] = Azeta
            self.data['Qzeta'] = Qzeta
            self.data['Jx']     = Jx
            self.data['Jy']     = Jy
            self.data['Jzeta']  = Jzeta
            self.data['Jcov']   = J_cov
            self.data['Jsmear'] = J_smear
            self.data['fsmear'] = f_smear
            self.data['fcov']   = f_cov
            #-------------------------
#===================================================














