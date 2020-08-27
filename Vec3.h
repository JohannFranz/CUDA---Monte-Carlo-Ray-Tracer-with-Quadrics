#ifndef VEC3_H
#define VEC3_H

#include <math.h>

class vec3
{
public:
	__device__				vec3() {}

	__device__				vec3(float a, float b, float c)
							{
								x = a;
								y = b;
								z = c;
							}

	__device__				vec3(const vec3& vec)
							{
								x = vec.x;
								y = vec.y;
								z = vec.z;
							}

	__device__ inline vec3	add(const vec3& v2) { return vec3(x + v2.x, y + v2.y, z + v2.z); }
	__device__ vec3			sub(const vec3& v2) { return vec3(x - v2.x, y - v2.y, z - v2.z); }
	__device__ inline vec3	mul(const vec3& v2) { return vec3(x * v2.x, y * v2.y, z * v2.z); }
	__device__ inline vec3	mul(float t) { return vec3(x * t, y * t, z * t); }
	__device__ inline vec3	div(float t) { return vec3(x / t, y / t, z / t); }
	
	__device__ inline float	dot(const vec3& v2) { return x * v2.x + y * v2.y + z * v2.z; }
	__device__ inline float	length() { return sqrtf(x*x + y * y + z * z); }
	__device__ inline vec3	normalize() { return vec3(x, y, z).div(length()); }
	__device__ vec3			cross(const vec3& v2)
							{
								return vec3((y * v2.z - z * v2.y), (-(x * v2.z - z * v2.x)), (x * v2.y - y * v2.x));
							}

	float x;
	float y;
	float z;
};

#endif