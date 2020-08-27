#include "WindowManager.h"

/*
Mapping from Cuda to OpenGL is based on "simpleCUDA2GL"-Code-Sample from NVidia
*/

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#pragma warning(disable:4996)
#endif
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <rendercheck_gl.h>
#include "WindowConstants.h"

WindowManager::WindowManager(unsigned int windowWidth, unsigned int windowHeight, unsigned int imageWidth, unsigned int imageHeight)
: _Window_Width(windowWidth), _Window_Height(windowHeight), _Image_Width(imageWidth), _Image_Height(imageHeight)
{
}

WindowManager::~WindowManager()
{
}

// display image to the screen as textured quad
void WindowManager::displayImage()
{
	glBindTexture(GL_TEXTURE_2D, _tex_cudaResult);
	glEnable(GL_TEXTURE_2D);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glViewport(0, 0, _Window_Width, _Window_Height);

	glUseProgram(_shDrawTex);
	GLint id = glGetUniformLocation(_shDrawTex, "cudaImage");
	glUniform1i(id, 0); // texture unit 0 to "cudaImage"
	SDK_CHECK_ERROR_GL();

	glBegin(GL_QUADS);
	glTexCoord2f(0.0, 0.0);
	glVertex3f(-1.0, -1.0, 0.5);
	glTexCoord2f(1.0, 0.0);
	glVertex3f(1.0, -1.0, 0.5);
	glTexCoord2f(1.0, 1.0);
	glVertex3f(1.0, 1.0, 0.5);
	glTexCoord2f(0.0, 1.0);
	glVertex3f(-1.0, 1.0, 0.5);
	glEnd();

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();

	glDisable(GL_TEXTURE_2D);
	glUseProgram(0);
	SDK_CHECK_ERROR_GL();
}

bool WindowManager::initGL(int *argc, char **argv)
{
	// Create GL context
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(_Window_Width, _Window_Height);
	_iGLUTWindowHandle = glutCreateWindow("CUDA OpenGL post-processing");

	// initialize necessary OpenGL extensions
	if (!isGLVersionSupported(2, 0) || !areGLExtensionsSupported(
			"GL_ARB_pixel_buffer_object "
			"GL_EXT_framebuffer_object"
		))
	{
		printf("ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		return false;
	}

	// default initialization
#ifndef USE_TEXTURE_RGBA8UI
	glClearColor(0.5, 0.5, 0.5, 1.0);
#else
	glClearColorIuiEXT(128, 128, 128, 255);
#endif
	glDisable(GL_DEPTH_TEST);

	// viewport
	glViewport(0, 0, _Window_Width, _Window_Height);

	// projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (GLfloat)_Window_Width / (GLfloat)_Window_Height, 0.1f, 10.0f);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	glEnable(GL_LIGHT0);
	float red[] = { 1.0f, 0.1f, 0.1f, 1.0f };
	float white[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, red);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, white);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 60.0f);

	SDK_CHECK_ERROR_GL();

	return true;
}

void WindowManager::initGLBuffers(struct cudaGraphicsResource** cuda_tex_resource)
{
	// create texture that will receive the result of CUDA
	createTextureDst(cuda_tex_resource, &_tex_cudaResult, _Image_Width, _Image_Height);
	// load shader programs
	_shDraw = compileGLSLprogram(NULL, WC::glsl_draw_fragshader_src);

	_shDrawTex = compileGLSLprogram(WC::glsl_drawtex_vertshader_src, WC::glsl_drawtex_fragshader_src);
	SDK_CHECK_ERROR_GL();
}

void WindowManager::createTextureDst(struct cudaGraphicsResource** cuda_tex_resource, GLuint *tex_cudaResult, unsigned int size_x, unsigned int size_y)
{
	// create a texture
	glGenTextures(1, tex_cudaResult);
	glBindTexture(GL_TEXTURE_2D, *tex_cudaResult);

	// set basic parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI_EXT, size_x, size_y, 0, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_BYTE, NULL);
	SDK_CHECK_ERROR_GL();
	// register this texture with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterImage(&(*cuda_tex_resource), *tex_cudaResult,
		GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));
}

void WindowManager::deleteTexture(GLuint *tex)
{
	glDeleteTextures(1, tex);
	SDK_CHECK_ERROR_GL();

	*tex = 0;
}

void WindowManager::closeWindow()
{
	deleteTexture(&_tex_screen);
	deleteTexture(&_tex_cudaResult);

	if (_iGLUTWindowHandle)
	{
		glutDestroyWindow(_iGLUTWindowHandle);
	}
}

GLuint WindowManager::compileGLSLprogram(const char *vertex_shader_src, const char *fragment_shader_src)
{
	GLuint v, f, p = 0;

	p = glCreateProgram();

	if (vertex_shader_src)
	{
		v = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(v, 1, &vertex_shader_src, NULL);
		glCompileShader(v);

		// check if shader compiled
		GLint compiled = 0;
		glGetShaderiv(v, GL_COMPILE_STATUS, &compiled);

		if (!compiled)
		{
			//#ifdef NV_REPORT_COMPILE_ERRORS
			char temp[256] = "";
			glGetShaderInfoLog(v, 256, NULL, temp);
			printf("Vtx Compile failed:\n%s\n", temp);
			//#endif
			glDeleteShader(v);
			return 0;
		}
		else
		{
			glAttachShader(p, v);
		}
	}

	if (fragment_shader_src)
	{
		f = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(f, 1, &fragment_shader_src, NULL);
		glCompileShader(f);

		// check if shader compiled
		GLint compiled = 0;
		glGetShaderiv(f, GL_COMPILE_STATUS, &compiled);

		if (!compiled)
		{
			//#ifdef NV_REPORT_COMPILE_ERRORS
			char temp[256] = "";
			glGetShaderInfoLog(f, 256, NULL, temp);
			printf("frag Compile failed:\n%s\n", temp);
			//#endif
			glDeleteShader(f);
			return 0;
		}
		else
		{
			glAttachShader(p, f);
		}
	}

	glLinkProgram(p);

	int infologLength = 0;
	int charsWritten = 0;

	glGetProgramiv(p, GL_INFO_LOG_LENGTH, (GLint *)&infologLength);

	if (infologLength > 0)
	{
		char *infoLog = (char *)malloc(infologLength);
		glGetProgramInfoLog(p, infologLength, (GLsizei *)&charsWritten, infoLog);
		printf("Shader compilation error: %s\n", infoLog);
		free(infoLog);
	}

	return p;
}



void WindowManager::reshape(int w, int h)
{
	_Window_Width = w;
	_Window_Height = h;
}

GLuint WindowManager::getCUDATexture()
{
	return _tex_cudaResult;
}

int	WindowManager::getWindowWidth()
{
	return _Window_Width;
}

int	WindowManager::getWindowHeight()
{
	return _Window_Height;
}

int WindowManager::getImageWidth()
{
	return _Image_Width;
}

int WindowManager::getImageHeight()
{
	return _Image_Height;
}