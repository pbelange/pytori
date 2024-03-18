import bpy
import json
from pathlib import Path


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
        # safekeeping old info
        old_mesh        = obj.data
        old_materials   = obj.data.materials.keys()
        
        # Overwriting
        obj.data = mesh
        for mat in old_materials:
            obj.data.materials.append(bpy.data.materials.get(mat))
        
        # Deleting old
        bpy.data.meshes.remove(old_mesh)
        
    obj.select_set(True)
    #================================
    
    return obj







data_file = './data/TORUS_001_X.json'


# LOAD FILE and create torus
#==========================
with open(bpy.path.abspath("//") + data_file , "r") as file: 
    mesh_data = json.load(file)

torus_name = Path(data_file).stem
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

#===========================
                
if 'r' in mesh_data['meta'].keys():
    bpy.ops.curve.primitive_bezier_circle_add(radius=mesh_data['meta']['r'], location=(0,0,0))
    co_curve = bpy.context.active_object
    co_curve.name = f'CoT_{torus_name}'
    co_curve.parent = object_2