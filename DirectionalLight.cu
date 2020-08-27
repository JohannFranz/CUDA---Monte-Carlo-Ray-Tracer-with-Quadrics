#ifndef DIRECTIONAL_LIGHT_CU
#define DIRECTIONAL_LIGHT_CU

#include "DirectionalLight.cuh"

__device__ DirectionalLight::DirectionalLight(vec3 color, vec3 direction, float intensity)
	:Light(color, intensity)
{
	_direction = direction;
}

__device__ DirectionalLight::~DirectionalLight()
{
}

__device__ vec3 DirectionalLight::getDirection()
{
	return _direction;
}

#endif