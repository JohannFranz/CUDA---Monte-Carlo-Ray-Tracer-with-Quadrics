#ifndef RAY_CU
#define RAY_CU

#include "Ray.h"
#include "RigidEntityList.h"
#include "RigidEntity.h"
#include "Hit.h"

__device__ Ray::Ray()
	: _start(vec4(0, 0, 0, 1)), _direction(vec4(0, 0, 0, 0))
{
}

__device__ Ray::Ray(vec4 start)
	: _start(start), _direction(vec4(0, 0, 0, 0))
{
}

__device__ Ray::Ray(vec4 start, vec4 direction)
	: _start(start), _direction(direction.normalize())
{
}

__device__ inline void Ray::setStart(vec4 newStart)
{ 
	_start = newStart; 
}

__device__ inline void Ray::setDirection(vec4 newDirection)
{
	_direction = newDirection.normalize();
}

__device__ inline const vec4 Ray::getStart() const
{ 
	return _start; 
}

__device__ inline const vec4 Ray::getDirection() const
{
	return _direction; 
}

__device__ inline Hit Ray::getHit() const
{
	return _hit;
}

//Return true if an object was hit, otherwise false
//The object that was hit will be saved in the ray as a hit member
__device__ bool Ray::trace(RigidEntityList& entityList)
{
	int countEntities = entityList.getNumEntities();
	RigidEntity** entities = entityList.getEntities();

	float t = FLT_MAX;
	vec4 position(0, 0, 0, 1);
	vec4 normal(0, 0, 0, 0);
	Material* mat;
	int id;

	//Loop over all entities
	for (int i = 0; i < countEntities; i++) {

		entities[i]->gotHit(*this, t, position, normal, &mat);
	}

	if (t == FLT_MAX) return false;

	_hit.setHitPosition(position, normal);
	_hit.setMaterial(mat);

	return true;
}
#endif