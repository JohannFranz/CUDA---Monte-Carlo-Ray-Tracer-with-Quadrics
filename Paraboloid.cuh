#ifndef PARABOLOID_CUH
#define PARABOLOID_CUH

#include "RigidEntity.h"
#include <math.h>
#include "Matrix4.h"

class Paraboloid : public RigidEntity
{
public:
	__device__	Paraboloid(matrix4& transform, Material* mat, float radius, float length);
	__device__	~Paraboloid();

	__device__ virtual bool	gotHit(Ray& ray, float& t, vec4& pos, vec4& normal, Material** mat) override;

};

__device__ Paraboloid::Paraboloid(matrix4& transform, Material* mat, float radius, float length)
{
	_mat = mat;
	_radius = radius;
	_halfLength = length * 0.5f;

	_coefficients.setIdentity();
	_coefficients._matrix[5] = 0;
	_coefficients._matrix[7] = -0.5f;
	_coefficients._matrix[13] = -0.5f;
	_coefficients._matrix[15] = 0;

	matrix4 transpose = transform.transpose();
	_coefficients = _coefficients.mulMat4BothSides(transpose, transform);

	_center = transform.getTranslation().mul(-1);
}

__device__ Paraboloid::~Paraboloid()
{
}

__device__ bool	Paraboloid::gotHit(Ray& ray, float& t, vec4& pos, vec4& normal, Material** mat)
{
	vec4 start = ray.getStart();
	vec4 dir = ray.getDirection();

	float uQu, sQs, uQs, root;
	float t1 = t;
	float t2 = t;

	if (quadricGotHit(start, dir, uQu, sQs, uQs, root) == false) return false;

	if (isRigidEntityInFront(root, t, t1, t2, uQu, uQs, sQs) == false) return false;

	float currentT = t;
	if (isHitpointInsideBounds(start, dir, pos, t1, t2, currentT) == false) return false;

	if (currentT >= t) return false;

	//At this point the ray hit a better hit point.
	t = currentT;

	pos = start.add(dir.mul(t));
	*mat = _mat;

	normal = getNormalOnQuadric(pos);
	if (normal.dot(dir) > 0) normal = normal.mul(-1.0f);
	return true;
}


#endif