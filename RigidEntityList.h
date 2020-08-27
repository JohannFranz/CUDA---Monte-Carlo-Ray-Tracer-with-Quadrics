#ifndef RIGID_ENTITY_LIST_H
#define RIGID_ENTITY_LIST_H

class MaterialList;
class RigidEntity;

class RigidEntityList {
public:
	__device__ RigidEntityList(MaterialList* materialListPtr);
	__device__ ~RigidEntityList();

	__device__ RigidEntity**	getEntities();
	__device__ unsigned int		getNumEntities();

private:
	RigidEntity**				_entities;
	unsigned int				_numEntities;
};


#endif