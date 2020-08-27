#ifndef CONE_CUH
#define CONE_CUH

#include "RigidEntity.h"
#include <math.h>
#include "Matrix4.h"

class Cone : public RigidEntity
{
public:
	__device__	Cone(matrix4& transform, Material* mat, float length);
	__device__	~Cone();

	__device__ virtual bool	gotHit(Ray& ray, float& t, vec4& pos, vec4& normal, Material** mat) override;

};

__device__ Cone::Cone(matrix4& transform, Material* mat, float length)
{
	_mat = mat;
	_halfLength = length * 0.5f;

	_coefficients.setIdentity();
	_coefficients._matrix[5] = -1.0f;
	_coefficients._matrix[15] = 0;

	matrix4 transpose = transform.transpose();
	_coefficients = _coefficients.mulMat4BothSides(transpose, transform);

	_center = transform.getTranslation().mul(-1);
}

__device__ Cone::~Cone()
{
}

__device__ bool	Cone::gotHit(Ray& ray, float& t, vec4& pos, vec4& normal, Material** mat)
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
	if (isHitpointInsideBounds(start, dir, pos, t1, t2, currentT) == false) return false;

	if (currentT >= t) return false;

	vec4 potentialHitpoint = start.add(dir.mul(currentT));
	if (potentialHitpoint.y > _center.y)
	{
		if (currentT == t1)
		{
			if (t2 > 0 && t2 > t1) currentT = t2;
			else return false;
		}
		else
		{
			if (t1 > 0 && t1 > t2) currentT = t1;
			else return false;
		}
		potentialHitpoint = start.add(dir.mul(currentT));
		if (potentialHitpoint.y > _center.y || potentialHitpoint.y < _center.y - _halfLength) return false;
	}
	if (currentT >= t) return false;

	//At this point the ray hit a better hit point.
	t = currentT;
	pos = start.add(dir.mul(t));
	*mat = _mat;
	normal = getNormalOnQuadric(pos);

	if (normal.dot(dir) > 0.0f)
	{
		normal.mul(-1.0f);
	}
	return true;
}

#endif