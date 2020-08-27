#ifndef LIGHTLIST_CU
#define LIGHTLIST_CU

#include "LightList.cuh"
#include "DirectionalLight.cu"

__device__ LightList::LightList()
	: _numDirectionalLights(3)
{
	//create array of Light-pointers
	_dirLights = (DirectionalLight**)malloc(sizeof(DirectionalLight*) * _numDirectionalLights);

	vec3 direction(-1, -1, -1);
	direction = direction.normalize();
	_dirLights[0] = new DirectionalLight(vec3(1,1,1), direction, 2.0f);

	direction = vec3(1, -1, -1);
	direction = direction.normalize();
	_dirLights[1] = new DirectionalLight(vec3(1, 1, 1), direction, 2.0f);

	/*direction = vec3(-1, 0.2f, 1);
	direction = direction.normalize();
	_dirLights[2] = new DirectionalLight(vec3(1, 1, 1), direction, 2.0f);*/

	direction = vec3(1, -0.2f, 1);
	direction = direction.normalize();
	_dirLights[2] = new DirectionalLight(vec3(1, 1, 1), direction, 1);
}

__device__ LightList::~LightList() {
	for (int i = 0; i < _numDirectionalLights; i++)
	{
		delete _dirLights[i];
	}
	free(_dirLights);
}

__device__ DirectionalLight** LightList::getDirectionalLights()
{
	return _dirLights;
}

__device__ int LightList::getNumDirectionalLights()
{
	return _numDirectionalLights;
}

#endif