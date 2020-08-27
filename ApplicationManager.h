#ifndef APP_MANAGER_H
#define APP_MANAGER_H

#include <helper_gl.h>
#include <GL/freeglut.h>

class WindowManager;
class StopWatchInterface;
class CheckRender;
class CudaParams;



class ApplicationManager
{
public:
	ApplicationManager();
	~ApplicationManager();

	void							run(int argc, char **argv);

	//glut-callback functions
	void							display();
	void							reshape(unsigned int width, unsigned int height);
	void							keyboard(unsigned char key, int x, int y);
	void							mouseButtonEvent(int button, int state, int x, int y);
	void							mouseMoveEvent(int x, int y);

	void							idle();
	void							mainMenu(int i);

	void							initGLContext(int argc, char **argv);
	void							initCUDAContext(int argc, char **argv);

private:
	void							initCudaRayTracingManager();
	void*							initRays();
	void							initCUDABuffers(unsigned int** texture, unsigned int** directLighting, unsigned int** indirectLighting);

	void							FreeResource();
	void							generateCUDAImage();
	void							Cleanup(int iExitCode);

	//void							printRandomNumbers(float* ranNumbers, int countNumbers);
	void							printTestSeeds(unsigned int amount);

private:
	WindowManager*					_WinMan;
	int								_FpsCount;
	int								_FpsLimit;
	StopWatchInterface*				_Timer;

	unsigned int					_Size_tex_data;
	unsigned int					_Num_texels;
	unsigned int					_Num_values;

	struct cudaGraphicsResource*	_Cuda_tex_result_resource;
	struct cudaGraphicsResource*	_Cuda_tex_screen_resource;
	
	// CheckFBO/BackBuffer class objects
	CheckRender*					_g_CheckRender;
	unsigned int					_g_TotalErrors;

	char*							_ref_file;
	bool							_enable_cuda;

	CudaParams*						_params;
};

#endif