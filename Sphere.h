#ifndef SPHERE_H
#define SPHERE_H

#include "RigidEntity.h"
#include <math.h>
#include "Matrix4.h"

class Sphere : public RigidEntity
{
public:
	__device__				Sphere(matrix4& transform, Material* mat);
	__device__				~Sphere();

	/*__device__ vec4			getCenter() { return _center; }
	__device__ float		getRadius() { return _radius; }*/

	//__device__ virtual bool	gotHit(Ray& ray, float& t, vec4& pos, vec4& normal) override;
	
private:
	/*vec4					_center;
	float					_radius;*/
};


__device__ Sphere::Sphere(matrix4& transform, Material* mat)
//:_center(center), _radius(radius)
{
	_mat = mat;
	
	_coefficients.setIdentity();
	_coefficients._matrix[15] = -1.0f;

	matrix4 transpose = transform.transpose();
	_coefficients = _coefficients.mulMat4BothSides(transpose, transform);
}

__device__ Sphere::~Sphere()
{
}
//
//__device__ bool	Sphere::gotHit(Ray& ray, float& t, vec4& pos, vec4& normal)
//{
//	vec4 rayStart = ray.getStart();
//	vec4 y = rayStart.sub(_center);
//	vec4 u = ray.getDirection();
//	float yy_dot = y.dot(y);
//	float yu_dot = y.dot(u);
//
//	float b = 2.0f * yu_dot;
//	float bsqr = b * b;
//	float c = yy_dot - _radius * _radius;
//
//	float root = bsqr - 4.0f * c;
//
//	if (root < 0.0f) return false;
//	root = sqrt(root);
//
//	float t1 = (-b + root) * 0.5f;
//	float t2 = (-b - root) * 0.5f;
//
//	if (t2 > t1) {
//		if (t1 > 0.0f) {
//			t = t1;
//		}
//	}
//	else {
//		if (t2 > 0.0f) {
//			t = t2;
//		}
//	}
//	pos = rayStart.add(u.mul(t));
//	normal = pos.sub(_center).normalize();
//
//	return true;
//}


#endif