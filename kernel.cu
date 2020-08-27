// Utilities and system includes
#include <helper_cuda.h>
#include "Vec4.h"
#include "CudaRayTracingManager.cu"


__global__ void concludeFrame(CudaParams params)
{
	CudaRayTracingManager* manager = *((CudaRayTracingManager**)params.rayTracingManager);

	manager->concludeFrame(params);
}

__global__ void prepareForNextFrame(CudaParams params)
{
	CudaRayTracingManager* manager = *((CudaRayTracingManager**)params.rayTracingManager);

	manager->prepareForNextFrame(params);
}

__global__ void processRayTracing(CudaParams params)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	CudaRayTracingManager* manager = *((CudaRayTracingManager**)params.rayTracingManager);
	SeedGenerator* seedGen = manager->getSeedGenerator();

	curandState_t randomState;
	curand_init(seedGen->getSeed(index), 0, 0, &randomState);

	vec3 rayColor;
	if (params.useAntiAliasing)
	{
		rayColor = manager->processRayTracingWithAntialiasing(params, randomState);
	}
	else
	{
		rayColor = manager->processRayTracing(params, randomState);
	}
	rayColor = rayColor.mul(255);
	params.g_odata[index] = Utils::rgbToInt(rayColor.x, rayColor.y, rayColor.z);
}

extern "C" void launch_cudaRayTracing(CudaParams& params)
{
	prepareForNextFrame << <1, params.threads >> > (params);
	processRayTracing<<<params.blocks, params.threads>>>(params);
	concludeFrame<< <1, 1 >> > (params);
}

__global__ void initRayTracingManager(void* manager, CudaParams params)
{
	//allocate space for the Manager.
	CudaRayTracingManager* managerPtr = new CudaRayTracingManager(params);

	//copy the address of the created managerPtr into the address the manager points to
	CudaRayTracingManager** helper = (CudaRayTracingManager**)manager;
	*helper = managerPtr;
}

//The seeds need to be randomly generated. Otherwise they would show a pattern after incrementing
__global__ void createRandomSeeds(CudaParams params)
{
	//int index = blockIdx.x*blockDim.x + threadIdx.x;
	int seedsPerBlock = (int)sqrtf(params.countRandomNumbers);
	int seedsPerThread = seedsPerBlock / blockDim.x;

	CudaRayTracingManager* manager = *((CudaRayTracingManager**)params.rayTracingManager);
	SeedGenerator* seedPtr = manager->getSeedGenerator();

	curandState_t state;
	curand_init(blockIdx.x, 0, seedsPerThread * threadIdx.x, &state);
	
	int start = blockIdx.x * seedsPerBlock + threadIdx.x * seedsPerThread;
	int end = start + seedsPerThread;

	for (int i = start; i < end; i++)
	{
		unsigned long long value = (unsigned long long)(curand_uniform(&state) * powf(10, 19));
		seedPtr->setSeed(i, value);
	}
}

__global__ void initRays(void* rayPtr, int numRays)
{
	//allocate space for all rays
	Ray* rays = (Ray*)malloc(numRays * sizeof(Ray));
	//create all Rays
	for (int i = 0; i < numRays; i++) {
		rays[i] = Ray();
	}
	//copy the address of the created rays into the address the rayPtr points to
	Ray** helper = (Ray**)rayPtr;
	*helper = rays;
}

//Depending on the job count a thread count will be calculated 
//The amount of threads is a proper divisor of the job amount
int getFittingThreads(int jobCount)
{
	int threadCount = 256;

	while (threadCount > 1)
	{
		if (jobCount % threadCount == 0) break;

		threadCount = (int) (threadCount * 0.5);
	}

	return threadCount;
}


void processSynchronize(cudaError_t& cudaStatus)
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

cudaError_t allocateGPUMem(void** ptr, unsigned int size)
{
	cudaError_t cudaStatus = cudaMalloc(ptr, sizeof(void*));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	return cudaStatus;
}

extern "C" void init_CudaRayTracingManager(CudaParams& params)
{
	cudaError_t cudaStatus = allocateGPUMem(&params.rayTracingManager, sizeof(void*));
	initRayTracingManager << <1, 1 >> > (params.rayTracingManager, params);
	
	unsigned int blocks = (unsigned int)sqrtf(params.countRandomNumbers);
	int threads = getFittingThreads(blocks);
	createRandomSeeds << <blocks, threads >> > (params);
	processSynchronize(cudaStatus);
}

extern "C" void init_CudaRays(void* rayPtr, int numRays)
{
	initRays<<<1,1>>>(rayPtr, numRays);
}

//#########################	Random Number Test ##############################
__global__ void fillRandomNumbers(unsigned long long* randomNumbers, CudaParams params)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	curandState_t state;
	curand_init(threadIdx.x, 0, blockDim.x, &state);
	for (int i = blockDim.x * threadIdx.x; i < blockDim.x * threadIdx.x + 4; i++)
	{
		randomNumbers[i] = (unsigned long long) (curand_uniform(&state) * powf(10, 19));
	}
}

extern "C" void testSeedGenerator(unsigned long long* randomNumbers, CudaParams& params)
{
	fillRandomNumbers << <1, 4 >> > (randomNumbers, params);
}
