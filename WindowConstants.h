#pragma once
#include <helper_gl.h>
#include <GL/freeglut.h>

/*
Mapping from Cuda to OpenGL is based on "simpleCUDA2GL"-Code-Sample from NVidia
*/

namespace WC{

	// constants / global variables
	const unsigned int REFRESH_DELAY	= 10;
	const unsigned int MAX_EPSILON		= 10;

	const GLenum fbo_targets[]			=
	{
		GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT,
		GL_COLOR_ATTACHMENT2_EXT, GL_COLOR_ATTACHMENT3_EXT
	};

	static const char *glsl_drawtex_vertshader_src =
		"void main(void)\n"
		"{\n"
		"	gl_Position = gl_Vertex;\n"
		"	gl_TexCoord[0].xy = gl_MultiTexCoord0.xy;\n"
		"}\n";

	static const char *glsl_drawtex_fragshader_src =
		"#version 130\n"
		"uniform usampler2D cudaImage;\n"
		"void main()\n"
		"{\n"
		"   vec4 c = texture(cudaImage, gl_TexCoord[0].xy);\n"
		"	gl_FragColor = c / 255.0;\n"
		"}\n";

	static const char *glsl_draw_fragshader_src =
		"#version 130\n"
		"out uvec4 FragColor;\n"
		"void main()\n"
		"{"
		"  FragColor = uvec4(gl_Color.xyz * 255.0, 255.0);\n"
		"}\n";


}
