#ifndef MATERIAL_LIST_H
#define MATERIAL_LIST_H

#include "Material.h"
#include "Vec3.h"

class MaterialList {
public:
	__device__ MaterialList();
	__device__ ~MaterialList();

	__device__ Material**	getMaterials();
	__device__ int			getNumMaterials();

private:
	Material**				_materials;
	unsigned int			_numMaterials;
};




#endif