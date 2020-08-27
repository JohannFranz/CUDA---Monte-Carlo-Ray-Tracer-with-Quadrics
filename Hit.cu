#include "Hit.h"

__device__ Hit::Hit()
	:_mat(NULL), _EntityID(-1)
{}

__device__ Hit::Hit(float distance, HitPosition hitPos, Material* mat/*, vec4 scatterDir*/)
	:_hitPos(hitPos), _mat(mat), _EntityID(-1)
{}

__device__ Hit::~Hit()
{}

__device__ inline HitPosition Hit::getHitPosition() const 
{ 
	return _hitPos; 
}

__device__ inline const Material* Hit::getMaterial() const 
{ 
	return _mat; 
}

__device__ inline int Hit::getEntityID() const 
{ 
	return _EntityID; 
}

__device__ inline void Hit::setEntityID(int id) 
{ 
	_EntityID = id; 
}

__device__ void Hit::setHitPosition(const HitPosition& hpos)
{
	setHitPosition(hpos.getPosition(), hpos.getNormal());
}

__device__ void	Hit::setHitPosition(const vec4& position, const vec4& normal)
{
	_hitPos.setNormal(normal);
	_hitPos.setPosition(position);
}

__device__ void Hit::setMaterial(const Material* mat)
{
	_mat = mat;
}

__device__ void	Hit::reset()
{
	_mat = NULL;
	_hitPos.reset();
}