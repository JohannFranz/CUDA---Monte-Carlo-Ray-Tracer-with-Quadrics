#ifndef BRDF_CUH
#define BRDF_CUH

#include "Vec3.h"
#include "Vec4.h"
#include "Constants.cuh"

//no ambient light implemented as it does not exist in reality
//BRDF and reflection formula based on phong-reflection model from "Ray Tracing from the Ground Up, p. 282"
class BRDF
{
public:
	__device__ BRDF(float diffuseFactor, float specularFactor, float shininess)
	{
		_kd = diffuseFactor;
		_ks = specularFactor;

		//check if brdf is energy conserving (kd + ks < 1)
		if (_kd + _ks > 1)
		{
			_kd = 0.5f;
			_ks = 0.49f;
		}

		_shininess = shininess;
	}

	__device__ vec3 getDiffuseBRDF(vec3 materialColor)
	{
		return materialColor.mul(_kd / M_PI);
	}

	//formula based from "Ray Tracing from the Ground Up, p. 282"
	__device__ float getSpecularBRDF(vec4 reflectionDir, vec4 viewDir)
	{
		return _ks * powf(reflectionDir.dot(viewDir), _shininess);
	}

private:
	float _kd;
	float _ks;
	float _shininess;
};

#endif