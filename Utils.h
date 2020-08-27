#pragma once

#include <stdlib.h>
#include "Vec3.h"
#include "Vec4.h"
#include "curand_kernel.h"
#include <helper_cuda.h>


namespace Utils
{
	__device__ float* randomNumbers;
	__device__ unsigned int randomNumAmount;

	// clamp x to range [a, b]
	__device__ float clamp(float x, float a, float b)
	{
		return max(a, min(b, x));
	}

	__device__ int clamp(int x, int a, int b)
	{
		return max(a, min(b, x));
	}

	// convert floating point rgb color to 8-bit integer
	__device__ int rgbToInt(float r, float g, float b)
	{
		r = clamp(r, 0.0f, 255.0f);
		g = clamp(g, 0.0f, 255.0f);
		b = clamp(b, 0.0f, 255.0f);
		return (int(b) << 16) | (int(g) << 8) | int(r);
	}

	__device__ vec3 intToRGB(int color)
	{
		vec3 rgb;
		rgb.x = (color & 0x000000FF);
		rgb.y = (color & 0x0000FF00) >> 8;
		rgb.z = (color & 0x00FF0000) >> 16;

		return rgb;
	}

	//creates a random number with seed
	__device__ void generateRandomBuffer()
	{
		int id = threadIdx.x + blockIdx.x * blockDim.x;
		curandState_t state;
		curand_init(0, id, 0, &state);

		randomNumbers[id] = curand_uniform(&state);
	}

	__device__ float getRandomNumber(int ranIndex)
	{
		return randomNumbers[ranIndex];
	}

	__device__ vec4 getReflectionVector(vec4 normal, vec4 in_dir)
	{
		vec4 reverseIndir = in_dir.mul(-1.0f);
		float projection = 2 * normal.dot(reverseIndir);
		normal = normal.mul(projection);
		vec4 reflection = normal.sub(reverseIndir);
		return reflection.normalize();
	}

	

	//creates a random number with seed
	/*__device__ float getRandomNumber(int ranIndex)
	{
		return randomNumbers[ranIndex%randomNumAmount];
	}*/

	//Idea from "Ray Tracing in a Weekend - Peter Shirley"
	/*__device__ vec4 random_in_unit_sphere(int ranIndex)
	{
		vec4 rand(getRandomNumber(ranIndex), getRandomNumber(ranIndex+1), getRandomNumber(ranIndex+2), 1);
		rand = rand.mul(2.0f);
		vec4 point = rand.sub(vec4(1, 1, 1, 1));
		return point;
	}*/
}

