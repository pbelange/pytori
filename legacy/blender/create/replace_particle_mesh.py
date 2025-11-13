import bpy
import numpy as np
from pathlib import Path
import json

# USER DEFINED PARAMETERS
#========================================================
data_path   = '/Users/pbelanger/ABPLocal/pytori/examples/henon4D/data'
data_file   = data_path + '/TORUS_001_X.json'


# LOAD FILE DATA
#----------------------
with open(bpy.path.abspath("//") + data_file , "r") as file: 
    torus_data = json.load(file)
torus_name = Path(data_file).stem
#----------------------


# List of particle coordinates
name      = torus_name
particles = torus_data['meta']['x-px']
#========================================================



def particles2mesh(particles,name):
    
    mesh_name       = f'mesh_{name}'

    # Creating mesh
    new_mesh = bpy.data.meshes.new(mesh_name)
    new_mesh.from_pydata(particles, edges = [], faces = [])
    new_mesh.update()
    
    
    return new_mesh


def replace_particle_mesh(particles,name):
    # Ensure an object is selected
    if bpy.context.selected_objects:
        # The object to replace the mesh of
        selected_object = bpy.context.selected_objects[0]
        
        # The new mesh data to use (replace 'NewMeshObjectName' with the actual name of your object)
        new_mesh_data = particles2mesh(particles,name)
        
        # Replace the mesh data
        old_mesh = selected_object.data
        selected_object.data = new_mesh_data
        bpy.data.meshes.remove(old_mesh)
    else:
        assert False, "Error: No object selected."
        
        
replace_particle_mesh(particles,name)