#ifndef LIGHTLIST_CUH
#define LIGHTLIST_CUH

class DirectionalLight;

class LightList
{
public:
	__device__ LightList();
	__device__ ~LightList();

	__device__ DirectionalLight**	getDirectionalLights();
	__device__ int					getNumDirectionalLights();

private:
	DirectionalLight**				_dirLights;
	unsigned int					_numDirectionalLights;
};

#endif
