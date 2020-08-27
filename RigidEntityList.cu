#include "RigidEntityList.h"
#include "MaterialList.h"
#include "RigidEntity.h"
#include "Matrix4.h"
#include "Sphere.cuh"
#include "Ellipsoid.cuh"
#include "Cylinder.cuh"
#include "Cone.cuh"
#include "Paraboloid.cuh"
#include "Hyperboloid.cuh"
#include "HyperboloidConnected.cuh"
#include "HyperbolicParaboloid.cuh"


__device__ RigidEntity** RigidEntityList::getEntities() 
{
	return _entities; 
}

__device__ unsigned int RigidEntityList::getNumEntities() 
{ 
	return _numEntities; 
}

__device__ RigidEntityList::RigidEntityList(MaterialList* materialList) : _numEntities(9)
{
	//create array of entity-pointers
	_entities = (RigidEntity**)malloc(sizeof(RigidEntity*) * _numEntities);
	Material** materials = materialList->getMaterials();

	matrix4 transform;
	transform.setIdentity();

	float a = 0.0000001f;
	float b = 1;
	float c = 0.0000001f;
	transform.setTranslation(vec3(0, 5, 0));
	_entities[0] = new Ellipsoid(transform, materials[0], a, b, c);

	float length = 10.0f;
	transform.setTranslation(vec3(-5, 0, -7));
	_entities[1] = new Cone(transform, materials[2], length);

	float radius = 3;
	transform.setTranslation(vec3(-16, 0, 0));
	_entities[2] = new Cylinder(transform, materials[4], radius, length);
	
	radius = 5;
	transform.setTranslation(vec3(0, 0, 0));
	_entities[3] = new Sphere(transform, materials[3], radius);

	transform.setTranslation(vec3(-8, 0, 5));
	_entities[4] = new Hyperboloid(transform, materials[2], radius, length);

	transform.setTranslation(vec3(8, 0, 6));
	_entities[5] = new HyperboloidConnected(transform, materials[4], radius, length);

	transform.setTranslation(vec3(8, 0, -6));
	_entities[6] = new Paraboloid(transform, materials[5], radius, length);

	transform.setTranslation(vec3(0, 0, 20));
	_entities[7] = new HyperbolicParaboloid(transform, materials[1], radius, length);

	a = 0.1f;
	b = 1;
	c = 0.1f;
	transform.setTranslation(vec3(5, 0, -15));
	_entities[8] = new Ellipsoid(transform, materials[6], a, b, c);
}

__device__ RigidEntityList::~RigidEntityList() {
	for (int i = 0; i < _numEntities; i++)
	{
		delete _entities[i];
	}
	free(_entities);
}