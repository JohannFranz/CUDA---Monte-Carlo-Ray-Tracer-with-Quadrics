#ifndef HIT_POSITION_H
#define HIT_POSITION_H

#include "Vec4.h"

class HitPosition
{
public:
	__device__						HitPosition()
									: _position(vec4(0, 0, 0, 1)), _normal(vec4(0, 0, 0, 0)) {}
	__device__						HitPosition(vec4 position, vec4 normal)
									: _position(position), _normal(normal.normalize()) {}

	__device__ inline void			setPosition(vec4 position) { _position = position; }
	__device__ inline void			setNormal(vec4 normal) { _normal = normal.normalize(); }

	__device__ inline vec4			getPosition() const { return _position; }
	__device__ inline vec4			getNormal() const { return _normal; }

	__device__ inline void			reset() {	_position = vec4(0, 0, 0, 1);
												_normal = vec4(0, 0, 0, 0);
											}

private:
	vec4	_position;	//the position where the ray hit a rigid body
	vec4	_normal;	//the normal on the rigid body
};

#endif