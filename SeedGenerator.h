#ifndef SEEDGENERATOR_H
#define SEEDGENERATOR_H

class SeedGenerator
{
public:
	__device__ SeedGenerator(unsigned int seedAmount);
	__device__ ~SeedGenerator();

	__device__ void					setSeed(int index, unsigned long long value);
	__device__ unsigned long long*	getSeeds();
	__device__ unsigned long long	getSeed(int index);

	__device__ void					incrementSeeds();


private:
	int					_seedAmount;
	unsigned long long*	_seeds;
};

#endif