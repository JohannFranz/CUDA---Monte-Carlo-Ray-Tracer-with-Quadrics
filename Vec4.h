#ifndef VEC4_H
#define VEC4_H

#include <math.h>
#include "Vec3.h"

class vec4
{
public:
	__device__				vec4() {}

	__device__				vec4(vec3 vec, float d)
	{
		x = vec.x;
		y = vec.y;
		z = vec.z;
		w = d;
	}

	__device__				vec4(float a, float b, float c, float d)
	{
		x = a;
		y = b;
		z = c;
		w = d;
	}

	__device__				vec4(const vec4& vec)
	{
		x = vec.x;
		y = vec.y;
		z = vec.z;
		w = vec.w;
	}

	__device__ inline vec4	add(const vec4& v2) const { return vec4(x + v2.x, y + v2.y, z + v2.z, w + v2.w); }
	__device__ vec4			sub(const vec4& v2) const { return vec4(x - v2.x, y - v2.y, z - v2.z, w - v2.w); }
	__device__ inline vec4	mul(const vec4& v2) const { return vec4(x * v2.x, y * v2.y, z * v2.z, w); }
	__device__ inline vec4	mul(float t) const { return vec4(x * t, y * t, z * t, w); }
	__device__ inline vec4	div(float t) const { return vec4(x / t, y / t, z / t, w / t); }

	__device__ inline vec3	toVec3() { return vec3(x, y, z); }
	__device__ inline float	dot(const vec4& v2) const { return x * v2.x + y * v2.y + z * v2.z; }
	__device__ inline float	length() const { return sqrtf(x*x + y * y + z * z); }
	__device__ vec4			normalize() const;
	__device__ vec4			cross(const vec4& v2) const
	{
		return vec4((y * v2.z - z * v2.y), (-(x * v2.z - z * v2.x)), (x * v2.y - y * v2.x), w);
	}

	float x;
	float y;
	float z;
	float w;
};

__device__ vec4	vec4::normalize() const
{ 
	float l = length();
	return vec4(x / l, y / l, z / l, w);
}

#endif