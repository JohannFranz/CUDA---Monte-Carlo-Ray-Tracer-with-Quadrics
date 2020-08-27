#ifndef RIGID_ENTITY_H
#define RIGID_ENTITY_H

#include "Ray.h"
#include "Material.h"
#include "Matrix4.h"

__device__ int	_ID = -1;


class RigidEntity
{
public:
	__device__ RigidEntity();

	__device__ virtual bool gotHit(Ray& ray, float& t, vec4& pos, vec4& normal, Material** mat) = 0;
	__device__ int			getId() { return _id; }


protected:
	__device__ bool			isHitpointInsideBounds(vec4& start, vec4& dir, vec4& pos, float& t1, float& t2, float& currentT);
	__device__ bool			isRigidEntityInFront(float& root, float& t, float& t1, float& t2, float& uQu, float& uQs, float& sQs);
	__device__ bool			quadricGotHit(vec4& start, vec4& dir, float& uQu, float& sQs, float& uQs, float& root);

	__device__ vec4			getNormalOnQuadric(vec4& pos);

protected:
	float		_halfLength;
	float		_radius;

	Material*	_mat;
	matrix4		_coefficients;
	vec4		_center;
	int			_id;
};

__device__ RigidEntity::RigidEntity()
{
	_ID += 1;
	_id = _ID;
}

__device__ vec4 RigidEntity::getNormalOnQuadric(vec4& pos)
{
	//calculate normal based on partial derivates of Q "An Introduction to Ray Tracing - p. 69"
	float xn = pos.x * _coefficients._matrix[0] + pos.y * _coefficients._matrix[4] + pos.z * _coefficients._matrix[8] + _coefficients._matrix[12];
	float yn = pos.x * _coefficients._matrix[1] + pos.y * _coefficients._matrix[5] + pos.z * _coefficients._matrix[9] + _coefficients._matrix[13];
	float zn = pos.x * _coefficients._matrix[2] + pos.y * _coefficients._matrix[6] + pos.z * _coefficients._matrix[10] + _coefficients._matrix[14];
	return vec4(xn, yn, zn, 0).normalize();
}

__device__ bool RigidEntity::quadricGotHit(vec4& start, vec4& dir, float& uQu, float& sQs, float& uQs, float& root)
{
	//a = uQu, b = uQs, c = sQs
	uQu = _coefficients.mulVec4BothSides(dir, dir);
	sQs = _coefficients.mulVec4BothSides(start, start);
	uQs = _coefficients.mulVec4BothSides(dir, start);

	//check if b² - 4ac is greater than 0
	root = uQs * uQs - uQu * sQs;
	return root >= 0.0f;
}

__device__ bool RigidEntity::isRigidEntityInFront(float& root, float& t, float& t1, float& t2, float& uQu, float& uQs, float& sQs)
{
	root = sqrtf(root);

	//Test if uQu is 0. Change the formular from "at² + bt + c = 0" to "bt + c = 0"
	//Solution: t = -c/b
	if (uQu == 0.0f) {
		t1 = -sQs / uQs;
	}
	else {
		t1 = (-uQs + root) / uQu;
		t2 = (-uQs - root) / uQu;
	}

	if (t1 >= t && t2 >= t) return false;

	return true;
}

__device__ bool RigidEntity::isHitpointInsideBounds(vec4& start, vec4& dir, vec4& pos, float& t1, float& t2, float& currentT)
{
	
	if (t1 < 0 && t2 < 0) return false;
	if (t1 > 0 && (t2 < 0 || t2 > t1))
	{
		currentT = t1;
		vec4 potentialHitpoint = start.add(dir.mul(currentT));
		if (potentialHitpoint.y > _center.y + _halfLength || potentialHitpoint.y < _center.y - _halfLength)
		{
			if (t2 < 0) return false;

			currentT = t2;
		}
		potentialHitpoint = start.add(dir.mul(currentT));
		if (potentialHitpoint.y > _center.y + _halfLength || potentialHitpoint.y < _center.y - _halfLength)
		{
			return false;
		}
	}
	else
	{
		currentT = t2;
		vec4 potentialHitpoint = start.add(dir.mul(currentT));
		if (potentialHitpoint.y > _center.y + _halfLength || potentialHitpoint.y < _center.y - _halfLength)
		{
			if (t1 < 0) return false;

			currentT = t1;
		}
		potentialHitpoint = start.add(dir.mul(currentT));
		if (potentialHitpoint.y > _center.y + _halfLength || potentialHitpoint.y < _center.y - _halfLength)
		{
			return false;
		}
	}

	return true;
}


#endif
