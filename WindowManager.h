#ifndef WINDOWMANAGERH
#define WINDOWMANAGERH

#include <helper_gl.h>
#include <GL/freeglut.h>
/*
Mapping from Cuda to OpenGL is based on "simpleCUDA2GL"-Code-Sample from NVidia
*/

class WindowManager {
public:
	WindowManager(	unsigned int windowWidth, unsigned int windowHeight, 
					unsigned int imageWidth, unsigned int imageHeight);
	~WindowManager();

	int				getWindowWidth();
	int				getWindowHeight();
	int				getImageWidth();
	int				getImageHeight();

	GLuint			getCUDATexture();

	// GL functionality
	bool			initGL(int *argc, char **argv);
	void			initGLBuffers(struct cudaGraphicsResource** cuda_tex_resource);

	void			displayImage();
	void			reshape(int w, int h);

	void			closeWindow();
	
private:
	GLuint			compileGLSLprogram(const char *vertex_shader_src, const char *fragment_shader_src);
	void			createTextureDst(struct cudaGraphicsResource** cuda_tex_resource, GLuint *tex_cudaResult, unsigned int size_x, unsigned int size_y);
	void			deleteTexture(GLuint *tex);

private:
	unsigned int	_Window_Width;
	unsigned int	_Window_Height;
	unsigned int	_Image_Width;
	unsigned int	_Image_Height;

	GLuint			_shDrawTex;  // draws a texture
	int				_iGLUTWindowHandle = 0;          // handle to the GLUT window
	GLuint			_fbo_source;

	// (offscreen) render target fbo variables
	GLuint			_tex_screen;      // where we render the image
	GLuint			_tex_cudaResult;  // where we will copy the CUDA result

	GLuint			_shDraw;
};


#endif