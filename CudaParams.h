#ifndef CUDA_PARAMS_H
#define CUDA_PARAMS_H

#include <helper_gl.h>
#include <GL/freeglut.h>

class CudaParams
{
public:
	unsigned int	threads;
	unsigned int	blocks;
	unsigned int*	g_odata; //the texture array for opengl display
	unsigned int*	directLighting; 
	unsigned int*	indirectLighting; 
	unsigned int	imgw;
	unsigned int	imgh;
	void*			rayTracingManager;
	unsigned int	countRandomNumbers;
	void*			primaryRays;
	bool			useAntiAliasing;

	int				mousePositionX;
	int				mousePositionY;

	/*	button: 0 - left mouse button
			1 - middle mouse button
			2 - right mouse button
			3 and 4 - scroll*/
	int				currentMouseButton;
	int				mouseState;
};

#endif