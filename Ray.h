#ifndef RAY_H
#define RAY_H

#include "Vec3.h"
#include "Vec4.h"
#include "CudaParams.h"
#include "Hit.h"

class HitPosition;
class Material;
class RigidEntityList;

class Ray {
public:

	__device__ Ray();
	__device__ Ray(vec4 start);
	__device__ Ray(vec4 start, vec4 direction);

	__device__ inline void			setStart(vec4 newStart);
	__device__ inline void			setDirection(vec4 newDirection);

	__device__ inline const vec4	getStart() const;
	__device__ inline const vec4	getDirection() const;
	__device__ inline Hit			getHit() const;

	__device__ bool					trace(RigidEntityList& entities);
	
private:
	vec4			_start;
	vec4			_direction;
	Hit				_hit;
};

#endif