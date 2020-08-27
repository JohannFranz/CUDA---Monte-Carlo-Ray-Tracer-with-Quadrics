#ifndef HIT_H
#define HIT_H

#include "Material.h"
#include "HitPosition.h"

class Hit {
public:
	__device__ Hit();
	__device__ Hit(float distance, HitPosition hitPos, Material* mat);
	__device__ ~Hit();

	__device__ inline HitPosition			getHitPosition() const;
	__device__ inline const  Material*		getMaterial() const;
	__device__ inline int					getEntityID() const;

	__device__ void							setHitPosition(const HitPosition& hpos);
	__device__ void							setHitPosition(const vec4& position, const vec4& normal);
	__device__ void							setMaterial(const Material* mat);
	__device__ inline void					setEntityID(int id);

	__device__ void							reset();

private:
	HitPosition		_hitPos;
	const Material*	_mat;
	int				_EntityID;
};



#endif