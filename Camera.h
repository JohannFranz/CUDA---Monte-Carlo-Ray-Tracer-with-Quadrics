#ifndef CAMERA_H
#define CAMERA_H


#include "Vec4.h"

class matrix4;

enum CameraAction
{
	none,
	rotate,
	scrollDown,
	scrollUp,
};

class Camera
{
public:
	__device__ Camera(vec4 eyePosition);
	
	__device__ void rotateCamera(matrix4& rotationMat);
	__device__ void rotateCamera(matrix4& coordSystem, matrix4& invCoordSystem, matrix4& rotationMat);
	__device__ void rotateVertically(float angle);

	__device__ vec4 getRayDirection(float imageWidth, float imageHeight);
	__device__ vec4 getRayDirection(float imageWidth, float imageHeight, curandState_t& randomState);
	__device__ vec4 getEyePosition();

	__device__ void setAction(CameraAction action, int posX, int posY);
	__device__ void	setAction(CameraAction action);

	__device__ void act(int mousePosX, int mousePosY);
		
private:
	__device__ void	moveCamera(vec4& moveDirection);

private:

	CameraAction	_action;
	int				_mousePosY;
	int				_mousePosX;

	vec4			_eyePosition;
	vec4			_upperLeftPos;
	vec4			_upperRightPos;
	vec4			_lowerLeftPos;
	vec4			_lowerRightPos;
};

#endif