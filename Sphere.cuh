#ifndef SPHERE_H
#define SPHERE_H

#include "RigidEntity.h"
#include <math.h>
#include "Matrix4.h"

class Sphere : public RigidEntity
{
public:
	
	__device__ Sphere(matrix4& transform, Material* mat, float radius);
	__device__ ~Sphere();

	__device__ virtual bool	gotHit(Ray& ray, float& t, vec4& pos, vec4& normal, Material** mat) override;
};

__device__ Sphere::Sphere(matrix4& transform, Material* mat, float radius)
{
	_mat = mat;

	_coefficients.setIdentity();
	_coefficients._matrix[15] = -1.0f;
	float matRadius = 1 / (radius * radius);

	_coefficients._matrix[0] = matRadius;
	_coefficients._matrix[5] = matRadius;
	_coefficients._matrix[10] = matRadius;

	matrix4 transpose = transform.transpose();
	_coefficients = _coefficients.mulMat4BothSides(transpose, transform);

	_center = transform.getTranslation().mul(-1);
}

__device__ Sphere::~Sphere() 
{
}

__device__ bool	Sphere::gotHit(Ray& ray, float& t, vec4& pos, vec4& normal, Material** mat)
{
	vec4 start = ray.getStart();
	vec4 dir = ray.getDirection();

	//a = uQu, b = uQs, c = sQs
	float uQu, sQs, uQs, root;
	float t1 = t;
	float t2 = t;

	if (quadricGotHit(start, dir, uQu, sQs, uQs, root) == false) return false;
	if (isRigidEntityInFront(root, t, t1, t2, uQu, uQs, sQs) == false) return false;

	float currentT = t;
	if (t2 > t1) {
		if (t1 > 0.0f) {
			currentT = t1;
		}
	}
	else {
		if (t2 > 0.0f) {
			currentT = t2;
		}
	}

	if (currentT >= t) return false;

	//At this point the ray hit a better hit point.
	t = currentT;

	pos = start.add(dir.mul(t));
	*mat = _mat;

	normal = getNormalOnQuadric(pos);
	return true;
}

#endif