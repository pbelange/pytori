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


# List of particle coordinates
name      = torus_name
particles = torus_data['meta']['x-px']
#========================================================








def vertex2instance_node_group(name):
	vertex2instance = bpy.data.node_groups.new(type = 'GeometryNodeTree', name = name)

	vertex2instance.is_modifier = True
	
	#initialize vertex2instance nodes
	#vertex2instance interface
	#Socket Geometry
	geometry_socket = vertex2instance.interface.new_socket(name = "Geometry", in_out='OUTPUT', socket_type = 'NodeSocketGeometry')
	geometry_socket.attribute_domain = 'POINT'
	
	#Socket Geometry
	geometry_socket_1 = vertex2instance.interface.new_socket(name = "Geometry", in_out='INPUT', socket_type = 'NodeSocketGeometry')
	geometry_socket_1.attribute_domain = 'POINT'
	
	
	#node Frame
	frame = vertex2instance.nodes.new("NodeFrame")
	frame.label = "Vertex2Points"
	frame.name = "Frame"
	frame.label_size = 20
	frame.shrink = True
	
	#node Frame.001
	frame_001 = vertex2instance.nodes.new("NodeFrame")
	frame_001.label = "Main Instance"
	frame_001.name = "Frame.001"
	frame_001.label_size = 20
	frame_001.shrink = True
	
	#node Group Input
	group_input = vertex2instance.nodes.new("NodeGroupInput")
	group_input.name = "Group Input"
	
	#node Store Named Attribute
	store_named_attribute = vertex2instance.nodes.new("GeometryNodeStoreNamedAttribute")
	store_named_attribute.name = "Store Named Attribute"
	store_named_attribute.data_type = 'FLOAT'
	store_named_attribute.domain = 'POINT'
	#Selection
	store_named_attribute.inputs[1].default_value = True
	#Name
	store_named_attribute.inputs[2].default_value = "Index"
	#Value_Vector
	store_named_attribute.inputs[3].default_value = (0.0, 0.0, 0.0)
	#Value_Color
	store_named_attribute.inputs[5].default_value = (0.0, 0.0, 0.0, 0.0)
	#Value_Bool
	store_named_attribute.inputs[6].default_value = False
	#Value_Int
	store_named_attribute.inputs[7].default_value = 0
	#Value_Rotation
	store_named_attribute.inputs[8].default_value = (0.0, 0.0, 0.0)
	
	#node Index
	index = vertex2instance.nodes.new("GeometryNodeInputIndex")
	index.name = "Index"
	
	#node Mesh to Points
	mesh_to_points = vertex2instance.nodes.new("GeometryNodeMeshToPoints")
	mesh_to_points.name = "Mesh to Points"
	mesh_to_points.mode = 'VERTICES'
	#Selection
	mesh_to_points.inputs[1].default_value = True
	#Position
	mesh_to_points.inputs[2].default_value = (0.0, 0.0, 0.0)
	#Radius
	mesh_to_points.inputs[3].default_value = 0.05000000074505806
	
	#node Instance on Points
	instance_on_points = vertex2instance.nodes.new("GeometryNodeInstanceOnPoints")
	instance_on_points.name = "Instance on Points"
	#Selection
	instance_on_points.inputs[1].default_value = True
	#Pick Instance
	instance_on_points.inputs[3].default_value = False
	#Instance Index
	instance_on_points.inputs[4].default_value = 0
	#Rotation
	instance_on_points.inputs[5].default_value = (0.0, 0.0, 0.0)
	#Scale
	instance_on_points.inputs[6].default_value = (1.0, 1.0, 1.0)
	
	#node Ico Sphere
	ico_sphere = vertex2instance.nodes.new("GeometryNodeMeshIcoSphere")
	ico_sphere.name = "Ico Sphere"
	#Subdivisions
	ico_sphere.inputs[1].default_value = 2
	
	#node Set Shade Smooth
	set_shade_smooth = vertex2instance.nodes.new("GeometryNodeSetShadeSmooth")
	set_shade_smooth.name = "Set Shade Smooth"
	set_shade_smooth.domain = 'FACE'
	#Selection
	set_shade_smooth.inputs[1].default_value = True
	#Shade Smooth
	set_shade_smooth.inputs[2].default_value = True
	
	#node Set Material
	set_material = vertex2instance.nodes.new("GeometryNodeSetMaterial")
	set_material.name = "Set Material"
	#Selection
	set_material.inputs[1].default_value = True
	if "Scan Index" in bpy.data.materials:
		set_material.inputs[2].default_value = bpy.data.materials["Scan Index"]
	
	#node Math
	math = vertex2instance.nodes.new("ShaderNodeMath")
	math.label = "Small radius"
	math.name = "Math"
	math.operation = 'DIVIDE'
	math.use_clamp = False
	#Value
	math.inputs[0].default_value = 1.0
	#Value_001
	math.inputs[1].default_value = 5.0
	#Value_002
	math.inputs[2].default_value = 0.5
	
	#node Reroute
	reroute = vertex2instance.nodes.new("NodeReroute")
	reroute.name = "Reroute"
	#node Realize Instances
	realize_instances = vertex2instance.nodes.new("GeometryNodeRealizeInstances")
	realize_instances.name = "Realize Instances"
	
	#node Group Output
	group_output = vertex2instance.nodes.new("NodeGroupOutput")
	group_output.name = "Group Output"
	group_output.is_active_output = True
	
	
	
	#Set parents
	group_input.parent = frame
	store_named_attribute.parent = frame
	index.parent = frame
	mesh_to_points.parent = frame
	ico_sphere.parent = frame_001
	set_shade_smooth.parent = frame_001
	set_material.parent = frame_001
	math.parent = frame_001
	
	#Set locations
	frame.location = (-236.83154296875, 305.0864562988281)
	frame_001.location = (-512.7505493164062, -80.79431915283203)
	group_input.location = (-118.3358154296875, -25.665924072265625)
	store_named_attribute.location = (-116.80621337890625, -105.08645629882812)
	index.location = (-266.90509033203125, -246.08834838867188)
	mesh_to_points.location = (52.36744689941406, -9.0169677734375)
	instance_on_points.location = (66.34434509277344, 321.0303039550781)
	ico_sphere.location = (41.706695556640625, -6.5670318603515625)
	set_shade_smooth.location = (187.44876098632812, -6.433662414550781)
	set_material.location = (332.8916931152344, -7.100883483886719)
	math.location = (-102.57989501953125, -6.893058776855469)
	reroute.location = (5.6748127937316895, -123.1544418334961)
	realize_instances.location = (230.1837158203125, 344.64984130859375)
	group_output.location = (389.42779541015625, 344.64984130859375)
	
	#Set dimensions
	frame.width, frame.height = 520.0, 358.0
	frame_001.width, frame_001.height = 635.0, 219.0
	group_input.width, group_input.height = 140.0, 100.0
	store_named_attribute.width, store_named_attribute.height = 140.0, 100.0
	index.width, index.height = 140.0, 100.0
	mesh_to_points.width, mesh_to_points.height = 140.0, 100.0
	instance_on_points.width, instance_on_points.height = 140.0, 100.0
	ico_sphere.width, ico_sphere.height = 140.0, 100.0
	set_shade_smooth.width, set_shade_smooth.height = 140.0, 100.0
	set_material.width, set_material.height = 140.0, 100.0
	math.width, math.height = 140.0, 100.0
	reroute.width, reroute.height = 16.0, 100.0
	realize_instances.width, realize_instances.height = 140.0, 100.0
	group_output.width, group_output.height = 140.0, 100.0
	
	#initialize vertex2instance links
	#math.Value -> ico_sphere.Radius
	vertex2instance.links.new(math.outputs[0], ico_sphere.inputs[0])
	#group_input.Geometry -> store_named_attribute.Geometry
	vertex2instance.links.new(group_input.outputs[0], store_named_attribute.inputs[0])
	#store_named_attribute.Geometry -> mesh_to_points.Mesh
	vertex2instance.links.new(store_named_attribute.outputs[0], mesh_to_points.inputs[0])
	#index.Index -> store_named_attribute.Value
	vertex2instance.links.new(index.outputs[0], store_named_attribute.inputs[4])
	#mesh_to_points.Points -> instance_on_points.Points
	vertex2instance.links.new(mesh_to_points.outputs[0], instance_on_points.inputs[0])
	#ico_sphere.Mesh -> set_shade_smooth.Geometry
	vertex2instance.links.new(ico_sphere.outputs[0], set_shade_smooth.inputs[0])
	#set_shade_smooth.Geometry -> set_material.Geometry
	vertex2instance.links.new(set_shade_smooth.outputs[0], set_material.inputs[0])
	#reroute.Output -> instance_on_points.Instance
	vertex2instance.links.new(reroute.outputs[0], instance_on_points.inputs[2])
	#instance_on_points.Instances -> realize_instances.Geometry
	vertex2instance.links.new(instance_on_points.outputs[0], realize_instances.inputs[0])
	#realize_instances.Geometry -> group_output.Geometry
	vertex2instance.links.new(realize_instances.outputs[0], group_output.inputs[0])
	#set_material.Geometry -> reroute.Input
	vertex2instance.links.new(set_material.outputs[0], reroute.inputs[0])
	return vertex2instance




def particles2instances(particles,name):
    
    # Init
    obj_name        = f'{name}'
    mesh_name       = f'mesh_{name}'
    material_name   = f'material_{name}'
    collection_name = f'scatter_{name}'
    geo_name        = f'geo_{name}'

    # Creating mesh
    new_mesh = bpy.data.meshes.new(mesh_name)
    new_mesh.from_pydata(particles, edges = [], faces = [])
    new_mesh.update()
    
    
    # make object from mesh
    new_object = bpy.data.objects.new(obj_name, new_mesh)
    # make collection
    new_collection = bpy.data.collections.new(collection_name)
    bpy.context.scene.collection.children.link(new_collection)
    
    # add object to scene collection
    new_collection.objects.link(new_object)
    
    # Select the object and make it active
    bpy.context.view_layer.objects.active = new_object
    new_object.select_set(True)
    
    # Creating geometry modifier:
    bpy.ops.object.mode_set(mode='OBJECT')
    
    new_object.modifiers.new('vertex2instance',type='NODES')
    new_object.modifiers['vertex2instance'].node_group = vertex2instance_node_group(name=obj_name)


    # Exiting
    bpy.context.view_layer.update()



    

particles2instances(particles,name)

