#ifndef MATERIAL_H
#define MATERIAL_H

#include "Vec3.h"
#include "BRDF.cuh"


class Material
{
public:

	__device__ Material(vec3 color, BRDF* brdf, bool isReflective)
	{
		_color = color;
		_brdf = brdf;
		_isReflective = isReflective;
	}
	
	__device__ BRDF*	getBRDF() const { return _brdf; }
	__device__ vec3		getColor() const { return _color; }
	__device__ bool		isReflective() const { return _isReflective; }
	
protected:
	bool	_isReflective;
	vec3	_color;
	BRDF*	_brdf;
};
#endif