import bpy
import numpy as np
from pathlib import Path
import json

# USER DEFINED PARAMETERS
#========================================================
data_path   = './data'
data_file   = data_path + '/TORUS_001_X.json'


# LOAD FILE DATA
#----------------------
with open(bpy.path.abspath("//") + data_file , "r") as file: 
    torus_data = json.load(file)
torus_name = Path(data_file).stem
#----------------------


# To pass to next functions
name      = torus_name
mesh_data = torus_data
#========================================================




def py2mesh(name,verts,faces,edges = []):
    #===============================
    # Create a new mesh:
    mesh = bpy.data.meshes.new(name+'_mesh')
    mesh.from_pydata(verts, edges, faces)
    mesh.update()
    #===============================


    #===============================
    # Linking to object
    obj = bpy.data.objects.get(name)
    if obj is None:
        # Create if None
        obj = bpy.data.objects.new(name,mesh)
        bpy.context.collection.objects.link(obj)
    else:
        assert False, "Error : Object already exists."
        
    obj.select_set(True)
    #================================
    
    return obj


def build_torus(mesh_data,name):
    torus_name = name
    obj = bpy.data.objects.get(name)
    assert obj is None, "Error : Object already exists."

    obj_in  = py2mesh(torus_name + '_in',   mesh_data['verts_in'],
                                        mesh_data['faces_in'])
    obj_out = py2mesh(torus_name,   mesh_data['verts_out'],
                                    mesh_data['faces_out'])
                                            
    # Deselect all objects
    bpy.ops.object.select_all(action='DESELECT')

    # Get the objects by name
    object_1 = bpy.data.objects[torus_name + '_in']
    object_2 = bpy.data.objects[torus_name]

    # Select the objects
    object_1.select_set(True)
    object_2.select_set(True)

    # Make one of them the active object (required for join operation)
    bpy.context.view_layer.objects.active = object_2


    # Join the objects
    bpy.ops.object.join()

    # Delete the old mesh
    bpy.data.meshes.remove(bpy.data.meshes[torus_name + '_in_mesh'])



    return object_2


def replace_torus_mesh(mesh_data,name):
    # Ensure an object is selected
    if bpy.context.selected_objects:
        # The object to replace the mesh of
        selected_object = bpy.context.selected_objects[0]
        
        # The new mesh data to use (replace 'NewMeshObjectName' with the actual name of your object)
        tmp_torus = build_torus(mesh_data,'tmp_'+name)
        
        # Replace the mesh data
        old_mesh = selected_object.data
        selected_object.data = tmp_torus.data
        selected_object.data.name = name
        bpy.data.meshes.remove(old_mesh)
        bpy.data.objects.remove(tmp_torus)



    else:
        assert False, "Error: No object selected."
        
        
replace_torus_mesh(mesh_data,name)