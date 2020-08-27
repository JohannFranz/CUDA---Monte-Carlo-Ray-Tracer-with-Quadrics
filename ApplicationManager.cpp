#include "ApplicationManager.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <rendercheck_gl.h>

#include "WindowManager.h"
#include "WindowConstants.h"
#include "CudaParams.h"

const char *sSDKname = "simpleCUDA2GL";
extern "C" void init_CudaRayTracingManager(CudaParams& params);
extern "C" void launch_cudaRayTracing(CudaParams& params);
extern "C" void init_CudaRays(void* rayPtr, int numRays);




ApplicationManager::ApplicationManager()
	: _FpsCount(0), _FpsLimit(1), _Timer(NULL), _g_CheckRender(NULL), _ref_file(NULL), _enable_cuda(true),
		_g_TotalErrors(0)
{
	_WinMan = new WindowManager(640, 640, 640, 640);
	fprintf(stdout, "1 << 29: %i\n", 1 << 29);
}

ApplicationManager::~ApplicationManager()
{
	delete _WinMan;
	delete _params;
	delete _g_CheckRender;
}

void ApplicationManager::initGLContext(int argc, char **argv)
{
	// First initialize OpenGL context, so we can properly set the GL for CUDA.
	// This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	if (false == _WinMan->initGL(&argc, argv))
	{
		return;
	}
}

void ApplicationManager::initCUDAContext(int argc, char **argv)
{
	// Now initialize CUDA context (GL context has been created already)
	//JF: This is a helper-function from "helper_cuda.h". It finds the fastest cuda device
	findCudaDevice(argc, (const char **)argv);

	sdkCreateTimer(&_Timer);
	sdkResetTimer(&_Timer);

	//JF: set Device-Heap size
	cudaError_t status = cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1 << 29);
	if (status != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d\n", status);
	}
}

void ApplicationManager::initCUDABuffers(unsigned int** texture, unsigned int** directLighting, unsigned int** indirectLighting)
{
	// set up vertex data parameter
	_Num_texels = _WinMan->getImageWidth() * _WinMan->getImageHeight();
	_Num_values = _Num_texels * 4;
	_Size_tex_data = sizeof(GLubyte) * _Num_values;
	checkCudaErrors(cudaMalloc((void **)&(*directLighting), _Size_tex_data));
	checkCudaErrors(cudaMalloc((void **)&(*texture), _Size_tex_data));
	checkCudaErrors(cudaMalloc((void **)&(*indirectLighting), _Size_tex_data));
}

void processCudaSynchronize(cudaError_t& cudaStatus)
{
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d\n", cudaStatus);
	}
}

extern "C" void testSeedGenerator(unsigned long long* randomNumbers, CudaParams& params);
void ApplicationManager::printTestSeeds(unsigned int amount)
{
	unsigned int size = sizeof(long long) * amount;

	//1. allocate GPU memory
	unsigned long long* randomNumbers;
	cudaError_t cudaStatus = cudaMalloc(&randomNumbers, size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	//2. create random vectors in GPU memory
	testSeedGenerator(randomNumbers, *_params);
	processCudaSynchronize(cudaStatus);

	//3. copy data to HOST memory
	unsigned long long* hostNumbers = (unsigned long long*)malloc(size);
	cudaStatus = cudaMemcpy(hostNumbers, randomNumbers, size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	processCudaSynchronize(cudaStatus);

	fprintf(stdout, "\n Random Numbers: \n");
	for (unsigned int i = 0; i < amount; i++) {
		fprintf(stdout, "%llu ", hostNumbers[i]);
		if (i%3 == 0 && i != 0) fprintf(stdout, "\n");
	}
}

void* ApplicationManager::initRays()
{
	void* rays;
	cudaError_t cudaStatus = cudaMalloc(&rays, sizeof(void*));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	int numRays = _WinMan->getImageWidth() * _WinMan->getImageHeight();
	init_CudaRays(rays, numRays);
	processCudaSynchronize(cudaStatus);
	return rays;
}

void ApplicationManager::initCudaRayTracingManager()
{
	_params->imgh = _WinMan->getImageHeight();
	_params->imgw = _WinMan->getImageWidth();
	int max = std::max(_params->imgw, _params->imgh);
	_params->countRandomNumbers = max * max; //needs to be square of int
	_params->threads = 128;
	_params->blocks = _params->imgh * _params->imgw / _params->threads;
	initCUDABuffers(&_params->g_odata, &_params->directLighting, &_params->indirectLighting);
	_params->primaryRays = initRays();

	init_CudaRayTracingManager(*_params);
	
	//############# Test Random Behaviour
	fprintf(stdout, "Vor printTestSeeds \n");
	printTestSeeds(17);
	fprintf(stdout, "Nach printTestSeeds \n");
}


 
void ApplicationManager::run(int argc, char **argv)
{
	_WinMan->initGLBuffers(&_Cuda_tex_result_resource);
	
	_params = new CudaParams();
	initCudaRayTracingManager();
	
	// Creating the Auto-Validation Code
	if (_ref_file)
	{
		_g_CheckRender = new CheckBackBuffer(_WinMan->getWindowWidth(), _WinMan->getWindowHeight(), 4);
		_g_CheckRender->setPixelFormat(GL_RGBA);
		_g_CheckRender->setExecPath(argv[0]);
		_g_CheckRender->EnableQAReadback(true);
	}

	printf("\n"
		"\tControls\n"
		"\t(right click mouse button for Menu)\n"
		"\t[esc] - Quit\n\n"
	);

	printf("%s Starting...\n\n", argv[0]);

	if (checkCmdLineFlag(argc, (const char **)argv, "file"))
	{

		getCmdLineArgumentString(argc, (const char **)argv, "file", &_ref_file);
	}

	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	if (checkCmdLineFlag(argc, (const char **)argv, "device"))
	{
		printf("[%s]\n", argv[0]);
		printf("   Does not explicitly support -device=n\n");
		printf("   This sample requires OpenGL.  Only -file=<reference> are supported\n");
		printf("exiting...\n");
		exit(EXIT_WAIVED);
	}


	// start rendering mainloop
	glutMainLoop();

	// Normally unused return path
	Cleanup(EXIT_SUCCESS);
}

void ApplicationManager::display()
{
	sdkStartTimer(&_Timer);

	if (_enable_cuda)
	{
		generateCUDAImage();
		_WinMan->displayImage();
	}

	cudaDeviceSynchronize();
	sdkStopTimer(&_Timer);

	// flip backbuffer
	glutSwapBuffers();

	// Update fps counter, fps/title display and log
	if (++_FpsCount == _FpsLimit)
	{
		char cTitle[256];
		float fps = 1000.0f / sdkGetAverageTimerValue(&_Timer);
		if (_params->useAntiAliasing)
		{
			sprintf(cTitle, "%.1f fps using Antialiasing", fps);
		}
		else {
			sprintf(cTitle, "%.1f fps", fps);
		}
		glutSetWindowTitle(cTitle);
		//printf("%s\n", cTitle);
		_FpsCount = 0;
		_FpsLimit = (int)((fps > 1.0f) ? fps : 1.0f);
		sdkResetTimer(&_Timer);
	}

	if (_params->currentMouseButton == 3 || _params->currentMouseButton == 4)
	{
		_params->currentMouseButton = -1;
	}
}

void ApplicationManager::reshape(unsigned int width, unsigned int height)
{
	_WinMan->reshape(width, height);
}

void ApplicationManager::generateCUDAImage()
{
	// execute CUDA kernel
	launch_cudaRayTracing(*_params);
	cudaDeviceSynchronize();
	// We want to copy cuda_dest_resource data to the texture
	// map buffer objects to get CUDA device pointers
	cudaArray *texture_ptr;
	checkCudaErrors(cudaGraphicsMapResources(1, &_Cuda_tex_result_resource, 0));
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, _Cuda_tex_result_resource, 0, 0));

	int num_texels = _WinMan->getImageWidth() * _WinMan->getImageHeight();
	int num_values = num_texels * 4;
	int size_tex_data = sizeof(GLubyte) * num_values;
	checkCudaErrors(cudaMemcpyToArray(texture_ptr, 0, 0, _params->g_odata, size_tex_data, cudaMemcpyDeviceToDevice));
	
	checkCudaErrors(cudaGraphicsUnmapResources(1, &_Cuda_tex_result_resource, 0));
}


void ApplicationManager::FreeResource()
{
	sdkDeleteTimer(&_Timer);

	// unregister this buffer object with CUDA
	//    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_tex_screen_resource));
#ifdef USE_TEXSUBIMAGE2D
	checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_dest_resource));
	deletePBO(&pbo_dest);
#else
	cudaFree(_params->g_odata);
	cudaFree(_params->directLighting);
	cudaFree(_params->indirectLighting);
#endif

	_WinMan->closeWindow();

	// finalize logs and leave
	printf("simpleCUDA2GL Exiting...\n");
}

void ApplicationManager::Cleanup(int iExitCode)
{
	FreeResource();
	printf("PPM Images are %s\n", (iExitCode == EXIT_SUCCESS) ? "Matching" : "Not Matching");
	exit(iExitCode);
}

////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void ApplicationManager::keyboard(unsigned char key, int /*x*/, int /*y*/)
{
	switch (key)
	{
	case('a'):
		_params->useAntiAliasing = !_params->useAntiAliasing;
		break;
	case (27):
		Cleanup(EXIT_SUCCESS);
		break;

	case ' ':
		_enable_cuda ^= 1;
#ifdef USE_TEXTURE_RGBA8UI

		if (enable_cuda)
		{
			glClearColorIuiEXT(128, 128, 128, 255);
		}
		else
		{
			glClearColor(0.5, 0.5, 0.5, 1.0);
		}

#endif
		break;

	}
}


/*	button: 0 - left mouse button
			1 - middle mouse button
			2 - right mouse button
			3 and 4 - scroll
	state:	0 - pressed
			1 - released
			*/
void ApplicationManager::mouseButtonEvent(int button, int state, int x, int y)
{
	// Wheel reports as button 3(scroll up) and button 4(scroll down)
	if ((button == 3) || (button == 4)) // It's a wheel event
	{
		// Each wheel event reports like a button click, GLUT_DOWN then GLUT_UP
		if (state == GLUT_UP) return; // Disregard redundant GLUT_UP events
		printf("Scroll %s At %d %d\n", (button == 3) ? "Up" : "Down", x, y);
	}
	else {  // normal button event
		printf("Button %s At %d %d\n", (state == GLUT_DOWN) ? "Down" : "Up", x, y);
	}

	//printf("Button, state %d %d \n", button, state);

	_params->currentMouseButton = button;
	_params->mouseState = state;
	_params->mousePositionX = x;
	_params->mousePositionY = y;
}

//function is repeatedly called as long as button is pressed
void ApplicationManager::mouseMoveEvent(int x, int y)
{
	_params->mousePositionX = x;
	_params->mousePositionY = y;
}

void ApplicationManager::idle()
{
	glutPostRedisplay();
}

void ApplicationManager::mainMenu(int i)
{
	keyboard((unsigned char)i, 0, 0);
}
