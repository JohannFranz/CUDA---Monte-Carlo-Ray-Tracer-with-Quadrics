#include "SeedGenerator.h"



__device__ SeedGenerator::SeedGenerator(unsigned int seedAmount)
{
	_seedAmount = seedAmount;
	_seeds = (unsigned long long*)malloc(sizeof(unsigned long long) * seedAmount);
}

__device__ SeedGenerator::~SeedGenerator()
{
	free(_seeds);
}

__device__ void SeedGenerator::setSeed(int index, unsigned long long value)
{
	_seeds[index] = value;
}

__device__ unsigned long long* SeedGenerator::getSeeds()
{
	return _seeds;
}

__device__ unsigned long long SeedGenerator::getSeed(int index)
{
	if (index > _seedAmount) return 0;
	
	return _seeds[index];
}

__device__ void SeedGenerator::incrementSeeds()
{
	int diff = _seedAmount / blockDim.x;
	int start = threadIdx.x * diff;
	int end = (threadIdx.x + 1) * diff;

	for (int i = start; i < end; i++)
	{
		_seeds[i] += 1;
	}
}
