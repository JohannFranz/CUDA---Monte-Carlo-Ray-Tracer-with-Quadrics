#ifndef CAMERA_CU
#define CAMERA_CU

#include "Camera.h"
#include "Matrix4.h"

__device__ Camera::Camera(vec4 eyePosition)
{
	_action = CameraAction::none;
	_eyePosition = eyePosition;

	vec4 lookAt(0, 0, -_eyePosition.z, 0);
	lookAt = lookAt.normalize();

	_upperLeftPos = _eyePosition.add(vec4(-1, 1, -1, 0));
	_upperRightPos = _eyePosition.add(vec4(1, 1, -1, 0));
	_lowerLeftPos = _eyePosition.add(vec4(-1, -1, -1, 0));
	_lowerRightPos = _eyePosition.add(vec4(1, -1, -1, 0));

	matrix4 rotationMatrix;
	rotationMatrix.setRotationX(-0.5f);
	rotateCamera(rotationMatrix);
}

__device__ void Camera::rotateCamera(matrix4& rotationMat)
{
	_eyePosition = rotationMat.mulVec4R(_eyePosition);
	_upperLeftPos = rotationMat.mulVec4R(_upperLeftPos);
	_upperRightPos = rotationMat.mulVec4R(_upperRightPos);
	_lowerLeftPos = rotationMat.mulVec4R(_lowerLeftPos);
	_lowerRightPos = rotationMat.mulVec4R(_lowerRightPos);
}

__device__ void Camera::rotateCamera(matrix4& coordSystem, matrix4& invCoordSystem, matrix4& rotationMat)
{
	_eyePosition = invCoordSystem.mulVec4R(_eyePosition);

	_eyePosition = rotationMat.mulVec4R(_eyePosition);

	_eyePosition = coordSystem.mulVec4R(_eyePosition);
}

__device__ void Camera::setAction(CameraAction action)
{
	_action = action;
}

__device__ void Camera::setAction(CameraAction action, int posX, int posY)
{
	if (_action == CameraAction::none && action != CameraAction::none)
	{
		_mousePosX = posX;
		_mousePosY = posY;
	}

	_action = action;
}

__device__ void Camera::rotateVertically(float angle)
{

	matrix4 rotationMatrix;
	rotationMatrix.setRotationX(angle);
	rotateCamera(rotationMatrix);
}

__device__ void Camera::act(int mousePosX, int mousePosY)
{
	if (_action == CameraAction::none) return;

	if (_action == CameraAction::rotate) {
		//rotate horizontal
		int mouseMovementX = _mousePosX - mousePosX;
		float angle = mouseMovementX / 100.0f;

		matrix4 rotationMatrix;
		rotationMatrix.setRotationY(angle);
		rotateCamera(rotationMatrix);

		_mousePosX = mousePosX;
		_mousePosY = mousePosY;
	}
	else if (_action == CameraAction::scrollDown) {
		vec4 moveDirection = _eyePosition.normalize();
		moveDirection.w = 0;
		moveCamera(moveDirection);
	}
	else if (_action == CameraAction::scrollUp) {
		vec4 moveDirection = _eyePosition.mul(-1).normalize();
		moveDirection.w = 0;
		moveCamera(moveDirection);
	}
}

__device__ void Camera::moveCamera(vec4& moveDirection)
{
	moveDirection = moveDirection.mul(2);
	_eyePosition = _eyePosition.add(moveDirection);
	_upperLeftPos = _upperLeftPos.add(moveDirection);
	_upperRightPos = _upperRightPos.add(moveDirection);
	_lowerLeftPos = _lowerLeftPos.add(moveDirection);
	_lowerRightPos = _lowerRightPos.add(moveDirection);
}

__device__ vec4 Camera::getEyePosition()
{
	return _eyePosition;
}

__device__ vec4 Camera::getRayDirection(float imageWidth, float imageHeight)
{
	int currentPixel = blockIdx.x*blockDim.x + threadIdx.x;
	int rows = 0;
	int column = 0;
	while (rows < imageHeight)
	{
		if (currentPixel < imageWidth)
		{
			column = currentPixel;
			break;
		}
		currentPixel -= imageWidth;
		rows += 1;
	}

	vec4 horizontal = _upperRightPos.sub(_upperLeftPos);
	vec4 vertical = _upperRightPos.sub(_lowerRightPos);

	float xCoord = (float)column / imageWidth;
	float yCoord = (float)rows / imageHeight;
	float sampleOffset = 1 / (imageWidth * 2);
	horizontal = horizontal.mul(xCoord + sampleOffset);
	vertical = vertical.mul(yCoord + sampleOffset);

	vec4 viewPoint = _lowerLeftPos.add(horizontal).add(vertical);

	return viewPoint.sub(_eyePosition);
}


__device__ vec4 Camera::getRayDirection(float imageWidth, float imageHeight, curandState_t& randomState)
{
	int currentPixel = blockIdx.x*blockDim.x + threadIdx.x;
	int rows = 0;
	int column = 0;
	while (rows < imageHeight)
	{
		if (currentPixel < imageWidth)
		{
			column = currentPixel;
			break;
		}
		currentPixel -= imageWidth;
		rows += 1;
	}

	vec4 horizontal = _upperRightPos.sub(_upperLeftPos);
	vec4 vertical = _upperRightPos.sub(_lowerRightPos);

	float xCoord = (float)column / imageWidth;
	float yCoord = (float)rows / imageHeight;
	float xOffset = curand_uniform(&randomState) / (imageWidth * 2);
	float yOffset = curand_uniform(&randomState) / (imageWidth * 2);
	horizontal = horizontal.mul(xCoord + xOffset);
	vertical = vertical.mul(yCoord + yOffset);

	vec4 viewPoint = _lowerLeftPos.add(horizontal).add(vertical);

	return viewPoint.sub(_eyePosition);
}

#endif