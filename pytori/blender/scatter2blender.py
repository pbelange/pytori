import bpy
import numpy as np
import json
from pathlib import Path


def TRANSFORM(sphere,empty_name,co_empty,phase_jumps):
    empty = bpy.data.objects.get(empty_name)
    
    # Apply Transformation constraint to the sphere
    constraint = sphere.constraints.new(type='TRANSFORM')
    constraint.target = empty
    constraint.target_space = 'CUSTOM'
    constraint.owner_space = 'CUSTOM'
    constraint.map_from = 'LOCATION'
    constraint.map_to = 'SCALE'
    
    # Set Custom Space (assuming co_empty is the object defining custom space)
    constraint.space_object = co_empty

    # Map from X location
    gap = (phase_jumps*2*np.pi) / 100
    constraint.from_min_x = -gap/2
    constraint.from_max_x =  gap/2

    # Map to scale
    constraint.map_to_x_from = 'X'
    constraint.map_to_y_from = 'X'
    constraint.map_to_z_from = 'X'
    
    constraint.to_min_x_scale = 0
    constraint.to_max_x_scale = 1
    constraint.to_min_y_scale = 0
    constraint.to_max_y_scale = 1
    constraint.to_min_z_scale = 0
    constraint.to_max_z_scale = 1

    # Mix mode
    constraint.mix_mode_scale = 'MULTIPLY'






data_file = './data/TORUS_001_X.json'


# LOAD FILE and create torus
#==========================
with open(bpy.path.abspath("//") + data_file , "r") as file: 
    torus_data = json.load(file)

torus_name = Path(data_file).stem

# List of particle coordinates
particles = torus_data['meta']['x-px']

# User input
collection_name = f"x-px_{torus_name}"
option = "wheel" 

# sphere_param
sphere_subdivisions = 2
sphere_radius = np.max(np.abs(particles)) / 100

# phase_jumps
phase_jumps = (0.25*100 / (len(particles) -1 ))
#===========================




# Ensure Blender is in Object Mode to prevent context errors
if bpy.context.active_object:
    if bpy.context.active_object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')





# Create or get the main collection
if collection_name not in bpy.data.collections:
    main_col = bpy.data.collections.new(collection_name)
    bpy.context.scene.collection.children.link(main_col)
else:
    main_col = bpy.data.collections[collection_name]


# Create or find a material named 'mat_MyExampleCollection'
material_name = f'mat_{collection_name}'
if material_name not in bpy.data.materials:
    mat = bpy.data.materials.new(name=material_name)
else:
    mat = bpy.data.materials[material_name]

# Enable 'Use Nodes' for the material
mat.use_nodes = True

# Create an empty at [0,0,0] and add it to the main collection
co_empty = bpy.data.objects.new(f'co_{collection_name}', None)
co_empty.location = (0, 0, 0)
main_col.objects.link(co_empty)

# Create utils collection and ensure it's properly nested
utils_col = bpy.data.collections.new(f'utils_{collection_name}')
main_col.children.link(utils_col)  # Properly link it as a subcollection of the main collection

# Create Bezier circle or curve and add it to the utils collection
if option == "wheel":
    bpy.ops.curve.primitive_bezier_circle_add(radius=1, location=(0,0,0))
elif option == "ruler":
    bpy.ops.curve.primitive_bezier_curve_add(location=(0,0,0))
    bpy.context.active_object.scale.z = len(particles) / 2  # Optional: adjust the curve's length

curve_obj = bpy.context.active_object
curve_obj.name = f'_idx_{option}_{collection_name}'
utils_col.objects.link(curve_obj)  # Move the curve object directly to the utils collection

# Ensure the curve object is not in any other collection
for col in curve_obj.users_collection:
    if col != utils_col:
        col.objects.unlink(curve_obj)


# Create and configure Empty objects for each particle, with follow path constraint
for idx in range(len(particles)):
    empty = bpy.data.objects.new(f'empty_{idx}', None)
    utils_col.objects.link(empty)  # Link to the utils subcollection
    constraint = empty.constraints.new(type='FOLLOW_PATH')
    constraint.target = curve_obj
    if option == "wheel":
        constraint.offset = -phase_jumps * (idx + 1) - 50
    elif option == "ruler":
        constraint.offset_factor = idx / (len(particles) - 1)
    empty.location = (0, 0, 0)  # Reset location to ensure proper positioning

# Create spheres at particle coordinates and ensure they're only in the main collection

# Create the master sphere at the origin
#=========================================
bpy.ops.mesh.primitive_ico_sphere_add(radius=sphere_radius, location= (0,0,0), subdivisions=sphere_subdivisions)
master_sphere = bpy.context.active_object
master_sphere.name = f'{collection_name}_master_sphere'

# Store the mesh data of the master sphere for use by other spheres
master_mesh_data = master_sphere.data

# Link the master sphere to the main collection and unlink from the context collection
main_col.objects.link(master_sphere)
for col in master_sphere.users_collection:
        if col != main_col:
            col.objects.unlink(master_sphere)
#=========================================

for idx, coord in enumerate(particles):
#    bpy.ops.mesh.primitive_ico_sphere_add(radius=sphere_radius, location=coord, subdivisions=sphere_subdivisions)
#    sphere = bpy.context.active_object
#    bpy.ops.object.shade_smooth()
#    sphere.name = f'{collection_name}_particle_{idx}'
    sphere = bpy.data.objects.new(f'{collection_name}_particle_{idx}', master_mesh_data)
    sphere.location = coord
    
    # Transform
    TRANSFORM(sphere,f'empty_{idx}',co_empty,phase_jumps)
    
    # Link sphere to the main collection and unlink from others
    main_col.objects.link(sphere)
    for col in sphere.users_collection:
        if col != main_col:
            col.objects.unlink(sphere)
            
            
# Setting the parent for curve_obj and all spheres to co_MyExampleCollection
curve_obj.parent = co_empty

# For each sphere, set co_MyExampleCollection as its parent
for particle_name in [f'{collection_name}_particle_{idx}' for idx in range(len(particles))]:
    obj = bpy.data.objects.get(particle_name)
    if obj:
        obj.parent = co_empty
        
        # Assigning material
        if len(obj.data.materials) == 0:
            # If the object has no materials, simply append the new one
            obj.data.materials.append(mat)
        else:
            # If the object already has materials, replace the first one
            obj.data.materials[0] = mat

