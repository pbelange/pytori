import bpy


mat = bpy.data.materials.new(name = "Cross Section")
mat.use_nodes = True
#initialize cross_section node group
def cross_section_node_group():

	cross_section = mat.node_tree
	#start with a clean node tree
	for node in cross_section.nodes:
		cross_section.nodes.remove(node)
	#cross_section interface
	
	#initialize cross_section nodes
	#node Frame.002
	frame_002 = cross_section.nodes.new("NodeFrame")
	frame_002.label = "OUTER SURFACE"
	frame_002.name = "Frame.002"
	frame_002.label_size = 20
	frame_002.shrink = True
	
	#node Frame.003
	frame_003 = cross_section.nodes.new("NodeFrame")
	frame_003.label = "Transparent  from x-y cuts"
	frame_003.name = "Frame.003"
	frame_003.label_size = 20
	frame_003.shrink = True
	
	#node Frame.001
	frame_001 = cross_section.nodes.new("NodeFrame")
	frame_001.label = "INNER GLOW"
	frame_001.name = "Frame.001"
	frame_001.label_size = 20
	frame_001.shrink = True
	
	#node Frame.004
	frame_004 = cross_section.nodes.new("NodeFrame")
	frame_004.label = "CLOSE TO PLANE GLOW"
	frame_004.name = "Frame.004"
	frame_004.label_size = 20
	frame_004.shrink = True
	
	#node Mix Shader.002
	mix_shader_002 = cross_section.nodes.new("ShaderNodeMixShader")
	mix_shader_002.name = "Mix Shader.002"
	#Fac
	mix_shader_002.inputs[0].default_value = 0.5
	
	#node Material Output.001
	material_output_001 = cross_section.nodes.new("ShaderNodeOutputMaterial")
	material_output_001.name = "Material Output.001"
	material_output_001.is_active_output = True
	material_output_001.target = 'ALL'
	#Displacement
	material_output_001.inputs[2].default_value = (0.0, 0.0, 0.0)
	#Thickness
	material_output_001.inputs[3].default_value = 0.0
	
	#node Mix Shader.003
	mix_shader_003 = cross_section.nodes.new("ShaderNodeMixShader")
	mix_shader_003.name = "Mix Shader.003"
	#Fac
	mix_shader_003.inputs[0].default_value = 0.5
	
	#node Mix Shader.001
	mix_shader_001 = cross_section.nodes.new("ShaderNodeMixShader")
	mix_shader_001.name = "Mix Shader.001"
	#Fac
	mix_shader_001.inputs[0].default_value = 0.5
	
	#node Separate XYZ.006
	separate_xyz_006 = cross_section.nodes.new("ShaderNodeSeparateXYZ")
	separate_xyz_006.name = "Separate XYZ.006"
	
	#node Math.010
	math_010 = cross_section.nodes.new("ShaderNodeMath")
	math_010.name = "Math.010"
	math_010.operation = 'MULTIPLY'
	math_010.use_clamp = False
	#Value_002
	math_010.inputs[2].default_value = 0.5
	
	#node Math.007
	math_007 = cross_section.nodes.new("ShaderNodeMath")
	math_007.name = "Math.007"
	math_007.operation = 'LESS_THAN'
	math_007.use_clamp = False
	#Value_002
	math_007.inputs[2].default_value = 0.5
	
	#node Math.008
	math_008 = cross_section.nodes.new("ShaderNodeMath")
	math_008.name = "Math.008"
	math_008.operation = 'LESS_THAN'
	math_008.use_clamp = False
	#Value_002
	math_008.inputs[2].default_value = 0.5
	
	#node Separate XYZ.007
	separate_xyz_007 = cross_section.nodes.new("ShaderNodeSeparateXYZ")
	separate_xyz_007.name = "Separate XYZ.007"
	
	#node Transparent BSDF.001
	transparent_bsdf_001 = cross_section.nodes.new("ShaderNodeBsdfTransparent")
	transparent_bsdf_001.name = "Transparent BSDF.001"
	#Color
	transparent_bsdf_001.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
	#Weight
	transparent_bsdf_001.inputs[1].default_value = 0.0
	
	#node Mix Shader.004
	mix_shader_004 = cross_section.nodes.new("ShaderNodeMixShader")
	mix_shader_004.name = "Mix Shader.004"
	
	#node Principled BSDF
	principled_bsdf = cross_section.nodes.new("ShaderNodeBsdfPrincipled")
	principled_bsdf.name = "Principled BSDF"
	principled_bsdf.distribution = 'MULTI_GGX'
	principled_bsdf.subsurface_method = 'RANDOM_WALK'
	#Base Color
	principled_bsdf.inputs[0].default_value = (0.12133609503507614, 0.34516558051109314, 0.7999992966651917, 1.0)
	#Metallic
	principled_bsdf.inputs[1].default_value = 0.0
	#Roughness
	principled_bsdf.inputs[2].default_value = 0.5
	#IOR
	principled_bsdf.inputs[3].default_value = 1.4500000476837158
	#Alpha
	principled_bsdf.inputs[4].default_value = 1.0
	#Normal
	principled_bsdf.inputs[5].default_value = (0.0, 0.0, 0.0)
	#Weight
	principled_bsdf.inputs[6].default_value = 0.0
	#Subsurface Weight
	principled_bsdf.inputs[7].default_value = 0.0
	#Subsurface Radius
	principled_bsdf.inputs[8].default_value = (1.0, 0.20000000298023224, 0.10000000149011612)
	#Subsurface Scale
	principled_bsdf.inputs[9].default_value = 0.05000000074505806
	#Subsurface IOR
	principled_bsdf.inputs[10].default_value = 1.399999976158142
	#Subsurface Anisotropy
	principled_bsdf.inputs[11].default_value = 0.0
	#Specular IOR Level
	principled_bsdf.inputs[12].default_value = 0.5
	#Specular Tint
	principled_bsdf.inputs[13].default_value = (1.0, 1.0, 1.0, 1.0)
	#Anisotropic
	principled_bsdf.inputs[14].default_value = 0.0
	#Anisotropic Rotation
	principled_bsdf.inputs[15].default_value = 0.0
	#Tangent
	principled_bsdf.inputs[16].default_value = (0.0, 0.0, 0.0)
	#Transmission Weight
	principled_bsdf.inputs[17].default_value = 0.0
	#Coat Weight
	principled_bsdf.inputs[18].default_value = 0.0
	#Coat Roughness
	principled_bsdf.inputs[19].default_value = 0.029999999329447746
	#Coat IOR
	principled_bsdf.inputs[20].default_value = 1.5
	#Coat Tint
	principled_bsdf.inputs[21].default_value = (1.0, 1.0, 1.0, 1.0)
	#Coat Normal
	principled_bsdf.inputs[22].default_value = (0.0, 0.0, 0.0)
	#Sheen Weight
	principled_bsdf.inputs[23].default_value = 0.0
	#Sheen Roughness
	principled_bsdf.inputs[24].default_value = 0.5
	#Sheen Tint
	principled_bsdf.inputs[25].default_value = (1.0, 1.0, 1.0, 1.0)
	#Emission Color
	principled_bsdf.inputs[26].default_value = (1.0, 1.0, 1.0, 1.0)
	#Emission Strength
	principled_bsdf.inputs[27].default_value = 0.0
	
	#node Texture Coordinate.007
	texture_coordinate_007 = cross_section.nodes.new("ShaderNodeTexCoord")
	texture_coordinate_007.label = "X-cut object"
	texture_coordinate_007.name = "Texture Coordinate.007"
	texture_coordinate_007.from_instancer = False
	if "x_cut" in bpy.data.objects:
		texture_coordinate_007.object = bpy.data.objects["x_cut"]
	
	#node Texture Coordinate.006
	texture_coordinate_006 = cross_section.nodes.new("ShaderNodeTexCoord")
	texture_coordinate_006.label = "Y-cut object"
	texture_coordinate_006.name = "Texture Coordinate.006"
	texture_coordinate_006.from_instancer = False
	if "y_cut" in bpy.data.objects:
		texture_coordinate_006.object = bpy.data.objects["y_cut"]
	
	#node Value
	value = cross_section.nodes.new("ShaderNodeValue")
	value.label = "ZERO"
	value.name = "Value"
	
	value.outputs[0].default_value = 0.0
	#node Mix Shader
	mix_shader = cross_section.nodes.new("ShaderNodeMixShader")
	mix_shader.name = "Mix Shader"
	
	#node Geometry.001
	geometry_001 = cross_section.nodes.new("ShaderNodeNewGeometry")
	geometry_001.name = "Geometry.001"
	
	#node Reroute
	reroute = cross_section.nodes.new("NodeReroute")
	reroute.name = "Reroute"
	#node Reroute.002
	reroute_002 = cross_section.nodes.new("NodeReroute")
	reroute_002.name = "Reroute.002"
	#node Reroute.003
	reroute_003 = cross_section.nodes.new("NodeReroute")
	reroute_003.name = "Reroute.003"
	#node Reroute.001
	reroute_001 = cross_section.nodes.new("NodeReroute")
	reroute_001.name = "Reroute.001"
	#node Emission
	emission = cross_section.nodes.new("ShaderNodeEmission")
	emission.name = "Emission"
	#Color
	emission.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
	#Weight
	emission.inputs[2].default_value = 0.0
	
	#node Value.002
	value_002 = cross_section.nodes.new("ShaderNodeValue")
	value_002.label = "INNER GLOW"
	value_002.name = "Value.002"
	
	value_002.outputs[0].default_value = 20.0
	#node Separate XYZ.010
	separate_xyz_010 = cross_section.nodes.new("ShaderNodeSeparateXYZ")
	separate_xyz_010.name = "Separate XYZ.010"
	
	#node Math.001
	math_001 = cross_section.nodes.new("ShaderNodeMath")
	math_001.name = "Math.001"
	math_001.operation = 'ABSOLUTE'
	math_001.use_clamp = False
	#Value_001
	math_001.inputs[1].default_value = 0.5
	#Value_002
	math_001.inputs[2].default_value = 0.5
	
	#node Math.002
	math_002 = cross_section.nodes.new("ShaderNodeMath")
	math_002.name = "Math.002"
	math_002.operation = 'LESS_THAN'
	math_002.use_clamp = False
	#Value_002
	math_002.inputs[2].default_value = 0.5
	
	#node Math.004
	math_004 = cross_section.nodes.new("ShaderNodeMath")
	math_004.name = "Math.004"
	math_004.operation = 'ABSOLUTE'
	math_004.use_clamp = False
	#Value_001
	math_004.inputs[1].default_value = 0.5
	#Value_002
	math_004.inputs[2].default_value = 0.5
	
	#node Math.005
	math_005 = cross_section.nodes.new("ShaderNodeMath")
	math_005.name = "Math.005"
	math_005.operation = 'LESS_THAN'
	math_005.use_clamp = False
	#Value_002
	math_005.inputs[2].default_value = 0.5
	
	#node Math.006
	math_006 = cross_section.nodes.new("ShaderNodeMath")
	math_006.name = "Math.006"
	math_006.operation = 'ADD'
	math_006.use_clamp = False
	#Value_002
	math_006.inputs[2].default_value = 0.5
	
	#node Separate XYZ.008
	separate_xyz_008 = cross_section.nodes.new("ShaderNodeSeparateXYZ")
	separate_xyz_008.name = "Separate XYZ.008"
	
	#node Math.003
	math_003 = cross_section.nodes.new("ShaderNodeMath")
	math_003.name = "Math.003"
	math_003.operation = 'GREATER_THAN'
	math_003.use_clamp = False
	#Value_002
	math_003.inputs[2].default_value = 0.5
	
	#node Math
	math = cross_section.nodes.new("ShaderNodeMath")
	math.name = "Math"
	math.operation = 'MULTIPLY'
	math.use_clamp = False
	#Value_002
	math.inputs[2].default_value = 0.5
	
	#node Math.011
	math_011 = cross_section.nodes.new("ShaderNodeMath")
	math_011.name = "Math.011"
	math_011.operation = 'GREATER_THAN'
	math_011.use_clamp = False
	#Value_002
	math_011.inputs[2].default_value = 0.5
	
	#node Emission.001
	emission_001 = cross_section.nodes.new("ShaderNodeEmission")
	emission_001.name = "Emission.001"
	#Color
	emission_001.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
	#Weight
	emission_001.inputs[2].default_value = 0.0
	
	#node Mix Shader.005
	mix_shader_005 = cross_section.nodes.new("ShaderNodeMixShader")
	mix_shader_005.name = "Mix Shader.005"
	
	#node Math.009
	math_009 = cross_section.nodes.new("ShaderNodeMath")
	math_009.name = "Math.009"
	math_009.operation = 'MULTIPLY'
	math_009.use_clamp = False
	#Value_002
	math_009.inputs[2].default_value = 0.5
	
	#node Texture Coordinate.008
	texture_coordinate_008 = cross_section.nodes.new("ShaderNodeTexCoord")
	texture_coordinate_008.label = "X-cut object"
	texture_coordinate_008.name = "Texture Coordinate.008"
	texture_coordinate_008.from_instancer = True
	if "x_cut" in bpy.data.objects:
		texture_coordinate_008.object = bpy.data.objects["x_cut"]
	
	#node Texture Coordinate.009
	texture_coordinate_009 = cross_section.nodes.new("ShaderNodeTexCoord")
	texture_coordinate_009.label = "Y-cut object"
	texture_coordinate_009.name = "Texture Coordinate.009"
	texture_coordinate_009.from_instancer = False
	if "y_cut" in bpy.data.objects:
		texture_coordinate_009.object = bpy.data.objects["y_cut"]
	
	#node Value.001
	value_001 = cross_section.nodes.new("ShaderNodeValue")
	value_001.label = "STATIC"
	value_001.name = "Value.001"
	
	value_001.outputs[0].default_value = -1.0
	#node Value.003
	value_003 = cross_section.nodes.new("ShaderNodeValue")
	value_003.label = "WIDTH"
	value_003.name = "Value.003"
	
	value_003.outputs[0].default_value = 0.5
	#node Value.004
	value_004 = cross_section.nodes.new("ShaderNodeValue")
	value_004.label = "GLOW"
	value_004.name = "Value.004"
	
	value_004.outputs[0].default_value = 20.0
	#Set parents
	separate_xyz_006.parent = frame_003
	math_010.parent = frame_003
	math_007.parent = frame_003
	math_008.parent = frame_003
	separate_xyz_007.parent = frame_003
	transparent_bsdf_001.parent = frame_003
	mix_shader_004.parent = frame_003
	principled_bsdf.parent = frame_002
	texture_coordinate_007.parent = frame_003
	texture_coordinate_006.parent = frame_003
	value.parent = frame_003
	mix_shader.parent = frame_001
	geometry_001.parent = frame_001
	reroute.parent = frame_001
	reroute_002.parent = frame_001
	reroute_003.parent = frame_001
	reroute_001.parent = frame_001
	emission.parent = frame_001
	value_002.parent = frame_001
	separate_xyz_010.parent = frame_004
	math_001.parent = frame_004
	math_002.parent = frame_004
	math_004.parent = frame_004
	math_005.parent = frame_004
	math_006.parent = frame_004
	separate_xyz_008.parent = frame_004
	math_003.parent = frame_004
	math.parent = frame_004
	math_011.parent = frame_004
	emission_001.parent = frame_004
	mix_shader_005.parent = frame_004
	math_009.parent = frame_004
	texture_coordinate_008.parent = frame_004
	texture_coordinate_009.parent = frame_004
	value_001.parent = frame_004
	value_003.parent = frame_004
	value_004.parent = frame_004
	
	#Set locations
	frame_002.location = (-334.65899658203125, 1115.3009033203125)
	frame_003.location = (-903.0, 628.0)
	frame_001.location = (101.0, 215.0)
	frame_004.location = (-1383.0, -332.0)
	mix_shader_002.location = (-60.0, 500.0)
	material_output_001.location = (740.0, 740.0)
	mix_shader_003.location = (400.0, 580.0)
	mix_shader_001.location = (180.0, 320.0)
	separate_xyz_006.location = (104.76513671875, -72.0096435546875)
	math_010.location = (442.6912841796875, -113.12423706054688)
	math_007.location = (278.83453369140625, -89.81207275390625)
	math_008.location = (281.71795654296875, -128.483154296875)
	separate_xyz_007.location = (103.4522705078125, -154.77218627929688)
	transparent_bsdf_001.location = (603.0, -88.0)
	mix_shader_004.location = (603.0, 52.0)
	principled_bsdf.location = (-65.34100341796875, 4.6990966796875)
	texture_coordinate_007.location = (-67.2137451171875, 111.8885498046875)
	texture_coordinate_006.location = (-64.97149658203125, -128.41757202148438)
	value.location = (283.0, -168.0)
	mix_shader.location = (-391.0, -35.0)
	geometry_001.location = (-811.0, -35.0)
	reroute.location = (-651.0, -95.0)
	reroute_002.location = (-611.0, -335.0)
	reroute_003.location = (-431.0, -335.0)
	reroute_001.location = (-611.0, -195.0)
	emission.location = (-589.0, -113.0)
	value_002.location = (-591.0, -235.0)
	separate_xyz_010.location = (103.0, -148.0)
	math_001.location = (323.0, 92.0)
	math_002.location = (483.0, 72.0)
	math_004.location = (323.0, -88.0)
	math_005.location = (483.0, -68.0)
	math_006.location = (663.0, 12.0)
	separate_xyz_008.location = (103.0, 32.0)
	math_003.location = (323.0, -168.0)
	math.location = (883.0, -68.0)
	math_011.location = (323.0, -308.0)
	emission_001.location = (883.0, -128.0)
	mix_shader_005.location = (1103.0001220703125, 52.0)
	math_009.location = (523.0, -228.0)
	texture_coordinate_008.location = (-67.2137451171875, 111.8885498046875)
	texture_coordinate_009.location = (-64.97149658203125, -128.41757202148438)
	value_001.location = (323.0, -208.0)
	value_003.location = (483.0, 32.0)
	value_004.location = (883.0, -248.0)
	
	#Set dimensions
	frame_002.width, frame_002.height = 300.0, 373.0
	frame_003.width, frame_003.height = 870.0, 541.0
	frame_001.width, frame_001.height = 620.0, 378.0
	frame_004.width, frame_004.height = 1370.0, 541.0
	mix_shader_002.width, mix_shader_002.height = 140.0, 100.0
	material_output_001.width, material_output_001.height = 140.0, 100.0
	mix_shader_003.width, mix_shader_003.height = 140.0, 100.0
	mix_shader_001.width, mix_shader_001.height = 140.0, 100.0
	separate_xyz_006.width, separate_xyz_006.height = 140.0, 100.0
	math_010.width, math_010.height = 140.0, 100.0
	math_007.width, math_007.height = 140.0, 100.0
	math_008.width, math_008.height = 140.0, 100.0
	separate_xyz_007.width, separate_xyz_007.height = 140.0, 100.0
	transparent_bsdf_001.width, transparent_bsdf_001.height = 140.0, 100.0
	mix_shader_004.width, mix_shader_004.height = 140.0, 100.0
	principled_bsdf.width, principled_bsdf.height = 240.0, 100.0
	texture_coordinate_007.width, texture_coordinate_007.height = 140.0, 100.0
	texture_coordinate_006.width, texture_coordinate_006.height = 140.0, 100.0
	value.width, value.height = 140.0, 100.0
	mix_shader.width, mix_shader.height = 140.0, 100.0
	geometry_001.width, geometry_001.height = 117.92095947265625, 100.0
	reroute.width, reroute.height = 16.0, 100.0
	reroute_002.width, reroute_002.height = 16.0, 100.0
	reroute_003.width, reroute_003.height = 16.0, 100.0
	reroute_001.width, reroute_001.height = 16.0, 100.0
	emission.width, emission.height = 140.0, 100.0
	value_002.width, value_002.height = 140.0, 100.0
	separate_xyz_010.width, separate_xyz_010.height = 140.0, 100.0
	math_001.width, math_001.height = 140.0, 100.0
	math_002.width, math_002.height = 140.0, 100.0
	math_004.width, math_004.height = 140.0, 100.0
	math_005.width, math_005.height = 140.0, 100.0
	math_006.width, math_006.height = 140.0, 100.0
	separate_xyz_008.width, separate_xyz_008.height = 140.0, 100.0
	math_003.width, math_003.height = 140.0, 100.0
	math.width, math.height = 140.0, 100.0
	math_011.width, math_011.height = 140.0, 100.0
	emission_001.width, emission_001.height = 140.0, 100.0
	mix_shader_005.width, mix_shader_005.height = 140.0, 100.0
	math_009.width, math_009.height = 140.0, 100.0
	texture_coordinate_008.width, texture_coordinate_008.height = 140.0, 100.0
	texture_coordinate_009.width, texture_coordinate_009.height = 140.0, 100.0
	value_001.width, value_001.height = 140.0, 100.0
	value_003.width, value_003.height = 140.0, 100.0
	value_004.width, value_004.height = 140.0, 100.0
	
	#initialize cross_section links
	#texture_coordinate_007.Object -> separate_xyz_006.Vector
	cross_section.links.new(texture_coordinate_007.outputs[3], separate_xyz_006.inputs[0])
	#texture_coordinate_006.Object -> separate_xyz_007.Vector
	cross_section.links.new(texture_coordinate_006.outputs[3], separate_xyz_007.inputs[0])
	#math_007.Value -> math_010.Value
	cross_section.links.new(math_007.outputs[0], math_010.inputs[0])
	#math_008.Value -> math_010.Value
	cross_section.links.new(math_008.outputs[0], math_010.inputs[1])
	#separate_xyz_006.X -> math_007.Value
	cross_section.links.new(separate_xyz_006.outputs[0], math_007.inputs[1])
	#separate_xyz_007.Y -> math_008.Value
	cross_section.links.new(separate_xyz_007.outputs[1], math_008.inputs[1])
	#value.Value -> math_008.Value
	cross_section.links.new(value.outputs[0], math_008.inputs[0])
	#value.Value -> math_007.Value
	cross_section.links.new(value.outputs[0], math_007.inputs[0])
	#math_010.Value -> mix_shader_004.Fac
	cross_section.links.new(math_010.outputs[0], mix_shader_004.inputs[0])
	#reroute.Output -> mix_shader.Fac
	cross_section.links.new(reroute.outputs[0], mix_shader.inputs[0])
	#transparent_bsdf_001.BSDF -> mix_shader_004.Shader
	cross_section.links.new(transparent_bsdf_001.outputs[0], mix_shader_004.inputs[2])
	#emission.Emission -> mix_shader.Shader
	cross_section.links.new(emission.outputs[0], mix_shader.inputs[2])
	#mix_shader.Shader -> mix_shader_002.Shader
	cross_section.links.new(mix_shader.outputs[0], mix_shader_002.inputs[2])
	#mix_shader_004.Shader -> mix_shader_002.Shader
	cross_section.links.new(mix_shader_004.outputs[0], mix_shader_002.inputs[1])
	#texture_coordinate_008.Object -> separate_xyz_008.Vector
	cross_section.links.new(texture_coordinate_008.outputs[3], separate_xyz_008.inputs[0])
	#math_001.Value -> math_002.Value
	cross_section.links.new(math_001.outputs[0], math_002.inputs[0])
	#value_003.Value -> math_002.Value
	cross_section.links.new(value_003.outputs[0], math_002.inputs[1])
	#math_004.Value -> math_005.Value
	cross_section.links.new(math_004.outputs[0], math_005.inputs[0])
	#texture_coordinate_009.Object -> separate_xyz_010.Vector
	cross_section.links.new(texture_coordinate_009.outputs[3], separate_xyz_010.inputs[0])
	#value_003.Value -> math_005.Value
	cross_section.links.new(value_003.outputs[0], math_005.inputs[1])
	#math_002.Value -> math_006.Value
	cross_section.links.new(math_002.outputs[0], math_006.inputs[0])
	#math_005.Value -> math_006.Value
	cross_section.links.new(math_005.outputs[0], math_006.inputs[1])
	#separate_xyz_008.X -> math_001.Value
	cross_section.links.new(separate_xyz_008.outputs[0], math_001.inputs[0])
	#separate_xyz_010.Y -> math_004.Value
	cross_section.links.new(separate_xyz_010.outputs[1], math_004.inputs[0])
	#math_003.Value -> math_009.Value
	cross_section.links.new(math_003.outputs[0], math_009.inputs[0])
	#separate_xyz_010.Y -> math_011.Value
	cross_section.links.new(separate_xyz_010.outputs[1], math_011.inputs[0])
	#separate_xyz_008.X -> math_003.Value
	cross_section.links.new(separate_xyz_008.outputs[0], math_003.inputs[0])
	#math_011.Value -> math_009.Value
	cross_section.links.new(math_011.outputs[0], math_009.inputs[1])
	#value_001.Value -> math_003.Value
	cross_section.links.new(value_001.outputs[0], math_003.inputs[1])
	#value_001.Value -> math_011.Value
	cross_section.links.new(value_001.outputs[0], math_011.inputs[1])
	#math_009.Value -> math.Value
	cross_section.links.new(math_009.outputs[0], math.inputs[1])
	#math_006.Value -> math.Value
	cross_section.links.new(math_006.outputs[0], math.inputs[0])
	#math.Value -> mix_shader_005.Fac
	cross_section.links.new(math.outputs[0], mix_shader_005.inputs[0])
	#mix_shader_003.Shader -> material_output_001.Surface
	cross_section.links.new(mix_shader_003.outputs[0], material_output_001.inputs[0])
	#mix_shader_005.Shader -> mix_shader_001.Shader
	cross_section.links.new(mix_shader_005.outputs[0], mix_shader_001.inputs[1])
	#mix_shader_002.Shader -> mix_shader_001.Shader
	cross_section.links.new(mix_shader_002.outputs[0], mix_shader_001.inputs[2])
	#emission_001.Emission -> mix_shader_005.Shader
	cross_section.links.new(emission_001.outputs[0], mix_shader_005.inputs[2])
	#principled_bsdf.BSDF -> mix_shader_003.Shader
	cross_section.links.new(principled_bsdf.outputs[0], mix_shader_003.inputs[1])
	#mix_shader_001.Shader -> mix_shader_003.Shader
	cross_section.links.new(mix_shader_001.outputs[0], mix_shader_003.inputs[2])
	#reroute_001.Output -> emission.Strength
	cross_section.links.new(reroute_001.outputs[0], emission.inputs[1])
	#geometry_001.Backfacing -> reroute.Input
	cross_section.links.new(geometry_001.outputs[6], reroute.inputs[0])
	#reroute_002.Output -> reroute_001.Input
	cross_section.links.new(reroute_002.outputs[0], reroute_001.inputs[0])
	#reroute_003.Output -> reroute_002.Input
	cross_section.links.new(reroute_003.outputs[0], reroute_002.inputs[0])
	#value_002.Value -> reroute_003.Input
	cross_section.links.new(value_002.outputs[0], reroute_003.inputs[0])
	#value_004.Value -> emission_001.Strength
	cross_section.links.new(value_004.outputs[0], emission_001.inputs[1])
	return cross_section

cross_section = cross_section_node_group()

