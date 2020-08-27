#ifndef CUDA_RAY_TRACING_MANAGER_H
#define CUDA_RAY_TRACING_MANAGER_H

#include "CudaParams.h"
#include "curand_kernel.h"
#include "Vec3.h"


class Ray;
class Hit;
class Camera;
class Material;
class MaterialList;
class RigidEntityList;
class SeedGenerator;
class LightList;
class HitPosition;

class CudaRayTracingManager
{
public:
	__device__ CudaRayTracingManager(CudaParams& params);
	__device__ ~CudaRayTracingManager();

	__device__ Ray				createRay(CudaParams& params, curandState_t& randomState);
	__device__ vec4				getPointOnHemisphere(Hit& hit, curandState_t& randomState);

	__device__ vec3				processRayTracing(CudaParams& params, curandState_t& randomState);
	__device__ vec3				processRayTracingWithAntialiasing(CudaParams& params, curandState_t& randomState);
	__device__ void				prepareForNextFrame(CudaParams& params);
	__device__ void				concludeFrame(CudaParams& params);

	__device__ SeedGenerator*	getSeedGenerator();
	__device__ int				getCountFrames();
	__device__ vec3				getLightAtPoint(HitPosition& hpos, const Material* mat, vec4 viewDir, vec4 reflectionDir);

private:
	__device__ vec3				getDirectLighting(CudaParams& params, curandState_t& randomState, Ray& primaryRay, Ray** primArray, Hit& firstHit);
	__device__ vec3				getIndirectLight(Ray& scatterRay, Hit& firstHit);
	__device__ Ray				getScatterRay(Ray& primaryRay, Hit& firstHit, curandState_t& randomState);
	
	__device__ void				clearBuffer(CudaParams& params);
	__device__ void				sumUpIndirectLight(vec3& indirectLight, CudaParams& params);


private:
	unsigned int		_imageWidth;
	unsigned int		_imageHeight;
	unsigned int		_frames;

	Camera*				_camera;
	MaterialList*		_materials;
	RigidEntityList*	_entities;
	LightList*			_lights;
	SeedGenerator*		_seedGen;

	bool				_useAntialiasing;
};

#endif