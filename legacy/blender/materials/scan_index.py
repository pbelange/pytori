
import bpy


mat = bpy.data.materials.new(name = "Scan Index")
mat.use_nodes = True
#initialize Scan Index node group
def scan_index_node_group():

	scan_index = mat.node_tree
	#start with a clean node tree
	for node in scan_index.nodes:
		scan_index.nodes.remove(node)
	#scan_index interface
	
	#initialize scan_index nodes
	#node Frame
	frame = scan_index.nodes.new("NodeFrame")
	frame.label = "Scan Index"
	frame.name = "Frame"
	frame.label_size = 20
	frame.shrink = True
	
	#node Value.001
	value_001 = scan_index.nodes.new("ShaderNodeValue")
	value_001.label = "Frame Scan"
	value_001.name = "Value.001"
	
	value_001.outputs[0].default_value = 8.09999942779541
	#node Value
	value = scan_index.nodes.new("ShaderNodeValue")
	value.label = "Scale indices"
	value.name = "Value"
	
	value.outputs[0].default_value = 1.0
	#node Attribute
	attribute = scan_index.nodes.new("ShaderNodeAttribute")
	attribute.name = "Attribute"
	attribute.attribute_name = "Index"
	attribute.attribute_type = 'GEOMETRY'
	
	#node Math.001
	math_001 = scan_index.nodes.new("ShaderNodeMath")
	math_001.name = "Math.001"
	math_001.operation = 'MULTIPLY'
	math_001.use_clamp = False
	#Value_002
	math_001.inputs[2].default_value = 0.5
	
	#node Math
	math = scan_index.nodes.new("ShaderNodeMath")
	math.name = "Math"
	math.operation = 'SUBTRACT'
	math.use_clamp = False
	#Value_002
	math.inputs[2].default_value = 0.5
	
	#node Color Ramp
	color_ramp = scan_index.nodes.new("ShaderNodeValToRGB")
	color_ramp.name = "Color Ramp"
	color_ramp.color_ramp.color_mode = 'RGB'
	color_ramp.color_ramp.hue_interpolation = 'NEAR'
	color_ramp.color_ramp.interpolation = 'LINEAR'
	
	#initialize color ramp elements
	color_ramp.color_ramp.elements.remove(color_ramp.color_ramp.elements[0])
	color_ramp_cre_0 = color_ramp.color_ramp.elements[0]
	color_ramp_cre_0.position = 0.0
	color_ramp_cre_0.alpha = 1.0
	color_ramp_cre_0.color = (0.0, 0.0, 0.0, 1.0)

	color_ramp_cre_1 = color_ramp.color_ramp.elements.new(1.0)
	color_ramp_cre_1.alpha = 1.0
	color_ramp_cre_1.color = (1.0, 1.0, 1.0, 1.0)

	
	#node Material Output
	material_output = scan_index.nodes.new("ShaderNodeOutputMaterial")
	material_output.name = "Material Output"
	material_output.is_active_output = True
	material_output.target = 'ALL'
	#Displacement
	material_output.inputs[2].default_value = (0.0, 0.0, 0.0)
	#Thickness
	material_output.inputs[3].default_value = 0.0
	
	#Set parents
	value_001.parent = frame
	value.parent = frame
	attribute.parent = frame
	math_001.parent = frame
	math.parent = frame
	
	#Set locations
	frame.location = (107.34260559082031, 116.15159606933594)
	value_001.location = (-47.18519592285156, -108.41365814208984)
	value.location = (-48.05900573730469, -30.60064697265625)
	attribute.location = (-193.6160888671875, -16.570556640625)
	math_001.location = (99.33784484863281, -33.22355651855469)
	math.location = (247.38572692871094, -33.70707702636719)
	color_ramp.location = (547.0752563476562, 238.5780487060547)
	material_output.location = (817.6461181640625, 258.8940124511719)
	
	#Set dimensions
	frame.width, frame.height = 641.0, 236.0
	value_001.width, value_001.height = 140.0, 100.0
	value.width, value.height = 140.0, 100.0
	attribute.width, attribute.height = 140.0, 100.0
	math_001.width, math_001.height = 140.0, 100.0
	math.width, math.height = 140.0, 100.0
	color_ramp.width, color_ramp.height = 240.0, 100.0
	material_output.width, material_output.height = 140.0, 100.0
	
	#initialize scan_index links
	#math_001.Value -> math.Value
	scan_index.links.new(math_001.outputs[0], math.inputs[0])
	#math.Value -> color_ramp.Fac
	scan_index.links.new(math.outputs[0], color_ramp.inputs[0])
	#color_ramp.Color -> material_output.Surface
	scan_index.links.new(color_ramp.outputs[0], material_output.inputs[0])
	#attribute.Fac -> math_001.Value
	scan_index.links.new(attribute.outputs[2], math_001.inputs[0])
	#value.Value -> math_001.Value
	scan_index.links.new(value.outputs[0], math_001.inputs[1])
	#value_001.Value -> math.Value
	scan_index.links.new(value_001.outputs[0], math.inputs[1])
	return scan_index

scan_index = scan_index_node_group()

