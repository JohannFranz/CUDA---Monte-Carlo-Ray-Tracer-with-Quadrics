#ifndef DIRECTIONAL_LIGHT_CUH
#define DIRECTIONAL_LIGHT_CUH

#include "Light.cuh"

class DirectionalLight : public Light
{
public:
	__device__ DirectionalLight(vec3 color, vec3 direction, float intensity);
	__device__ ~DirectionalLight();

	__device__ vec3 getDirection();

private:
	vec3 _direction;
};

#endif