#ifndef SPHERE_H
#define SPHERE_H

#include "RigidEntity.h"
#include <math.h>

class Ellipsoid : public RigidEntity
{
public:
	__device__				Ellipsoid(vec4& center, vec4& stretch, float radius, Material* mat);
	__device__				~Ellipsoid();

	__device__ vec4			getCenter() { return _center; }
	__device__ float		getRadius() { return _radius; }

	//__device__ virtual bool	gotHit(Ray& ray, float& t, vec4& pos, vec4& normal) override;

private:
	vec4					_center;
	float					_radius;
};

__device__ Ellipsoid::Ellipsoid(vec4& center, vec4& stretch, float radius, Material* mat)
	:_center(center), _radius(radius)
{
	_mat = mat;
	_coefficients._matrix[0] = stretch.x;
	_coefficients._matrix[5] = stretch.y;
	_coefficients._matrix[10] = stretch.z;
	_coefficients._matrix[15] = -1.0f;
}

__device__ Ellipsoid::~Ellipsoid()
{
}

#endif