#ifndef LIGHT_CUH
#define LIGHT_CUH

#include "Vec3.h"


class Light
{
public:
	__device__ Light(vec3 color, float intensity)
		:_color(color), _intensity(intensity)
	{
		_isActive = true;
	}

	__device__ void		setIntensity(float intensity) { _intensity = intensity; }

	__device__ vec3		getColor() { return _color; }
	__device__ float	getIntensity() { return _intensity; }
	__device__ bool		isActive() { return _isActive; }

protected:
	
	vec3	_color;
	float	_intensity;
	bool	_isActive;
};

#endif