#ifndef HYPERBOLIC_PARABOLOID_CUH
#define HYPERBOLIC_PARABOLOID_CUH

#include "RigidEntity.h"
#include <math.h>
#include "Matrix4.h"

class HyperbolicParaboloid : public RigidEntity
{
public:
	__device__	HyperbolicParaboloid(matrix4& transform, Material* mat, float radius, float length);
	__device__	~HyperbolicParaboloid();

	__device__ virtual bool	gotHit(Ray& ray, float& t, vec4& pos, vec4& normal, Material** mat) override;

};

__device__ HyperbolicParaboloid::HyperbolicParaboloid(matrix4& transform, Material* mat, float radius, float length)
{
	_mat = mat;
	_radius = radius;
	_halfLength = length * 0.5f;

	_coefficients.setIdentity();
	_coefficients._matrix[0] = 0.5f;
	_coefficients._matrix[5] = 0;
	_coefficients._matrix[7] = -0.5f * 10;
	_coefficients._matrix[10] = -1;
	_coefficients._matrix[13] = -0.5f * 10;
	_coefficients._matrix[15] = 0;

	matrix4 transpose = transform.transpose();
	_coefficients = _coefficients.mulMat4BothSides(transpose, transform);

	_center = transform.getTranslation().mul(-1);
}

__device__ HyperbolicParaboloid::~HyperbolicParaboloid()
{
}

__device__ bool	HyperbolicParaboloid::gotHit(Ray& ray, float& t, vec4& pos, vec4& normal, Material** mat)
{
	vec4 start = ray.getStart();
	vec4 dir = ray.getDirection();

	float uQu, sQs, uQs, root;
	float t1 = t;
	float t2 = t;

	if (quadricGotHit(start, dir, uQu, sQs, uQs, root) == false) return false;

	if (isRigidEntityInFront(root, t, t1, t2, uQu, uQs, sQs) == false) return false;

	float currentT = t;
	/*if (isHitpointInsideBounds(start, dir, pos, t1, t2, currentT) == false) return false;*/
	if (t1 < 0 && t2 < 0) return false;
	if (t1 > 0 && (t2 < 0 || t2 > t1))
	{
		currentT = t1;
		vec4 potentialHitpoint = start.add(dir.mul(currentT));
		if (	potentialHitpoint.x > _center.x + _halfLength 
			||	potentialHitpoint.x < _center.x - _halfLength
			||	potentialHitpoint.y > _center.y + _halfLength 
			||	potentialHitpoint.y < _center.y - _halfLength)
		{
			if (t2 < 0) return false;

			currentT = t2;
		}
		potentialHitpoint = start.add(dir.mul(currentT));
		if (	potentialHitpoint.x > _center.x + _halfLength
			||	potentialHitpoint.x < _center.x - _halfLength
			||	potentialHitpoint.y > _center.y + _halfLength
			||	potentialHitpoint.y < _center.y - _halfLength)
		{
			return false;
		}
	}
	else
	{
		currentT = t2;
		vec4 potentialHitpoint = start.add(dir.mul(currentT));
		if (	potentialHitpoint.x > _center.x + _halfLength
			||	potentialHitpoint.x < _center.x - _halfLength
			||	potentialHitpoint.y > _center.y + _halfLength
			||	potentialHitpoint.y < _center.y - _halfLength)
		{
			if (t1 < 0) return false;

			currentT = t1;
		}
		potentialHitpoint = start.add(dir.mul(currentT));
		if (	potentialHitpoint.x > _center.x + _halfLength
			||	potentialHitpoint.x < _center.x - _halfLength
			||	potentialHitpoint.y > _center.y + _halfLength
			||	potentialHitpoint.y < _center.y - _halfLength)
		{
			return false;
		}
	}

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