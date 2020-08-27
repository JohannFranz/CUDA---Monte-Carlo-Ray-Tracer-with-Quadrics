#include "MaterialList.h"
#include "BRDF.cuh"

__device__ Material** MaterialList::getMaterials() 
{ 
	return _materials; 
}

__device__ int MaterialList::getNumMaterials() 
{ 
	return _numMaterials; 
}

__device__ MaterialList::MaterialList() 
	: _numMaterials(6)
{
	//create array of material-pointers
	_materials = (Material**)malloc(sizeof(Material*) * _numMaterials);

	//green lambertian material
	vec3 color(0, 1, 0);
	BRDF* greenLambertian = new BRDF(0.99f, 0, 0);
	_materials[0] = new Material(color, greenLambertian, false);

	//black shiny material
	color = vec3(0, 0, 0);
	BRDF* blackShiny = new BRDF(0, 0.99f, 100);
	_materials[1] = new Material(color, blackShiny, false);

	//lilac shiny material
	color = vec3(1, 0, 1);
	BRDF* lilacShiny = new BRDF(0.6f, 0.39f, 100);
	_materials[2] = new Material(color, lilacShiny, false);

	//mirror material
	color = vec3(0.99f, 0.99f, 0.99f);
	//BRDF* shiny = new BRDF(0.99f, 0, 0);
	BRDF* mirror = new BRDF(0, 0.99f, 100);
	_materials[3] = new Material(color, mirror, true);

	//blue lambertian material
	color = vec3(0, 0, 1);
	BRDF* blueLambertian = new BRDF(0.99f, 0, 0);
	_materials[4] = new Material(color, blueLambertian, false);

	//red lambertian material
	color = vec3(1, 0, 0);
	BRDF* redLambertian = new BRDF(0.99f, 0, 0);
	_materials[5] = new Material(color, redLambertian, false);

	//magenta shiny material
	color = vec3(1, 1, 0);
	BRDF* magentaShiny = new BRDF(0.4f, 0.59f, 100);
	_materials[6] = new Material(color, magentaShiny, false);
}

__device__ MaterialList::~MaterialList() {
	for (int i = 0; i < _numMaterials; i++)
	{
		delete _materials[i];
	}
	free(_materials);
}