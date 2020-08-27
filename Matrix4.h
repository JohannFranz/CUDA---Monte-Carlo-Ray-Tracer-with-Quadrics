#ifndef MATRIX4_H
#define MATRIX4_H

#include <math.h>
#include "Vec4.h"

//The matrix is a 1D-Array, columns based
class matrix4
{
public:
	__device__	matrix4();
	__device__	matrix4(const vec4& col1, const vec4& col2, const vec4& col3, const vec4& col4);

	__device__ void		setIdentity();

	__device__ void		setProjectionRow(float a, float b, float c, float d);
	__device__ void		setTranslation(const vec3& trans);
	__device__ void		setRotationX(float angle);
	__device__ void		setRotationY(float angle);
	__device__ void		setRotationZ(float angle);

	__device__ vec4		getTranslation() const;

	//multipy matrix by a vector from the right side
	__device__	vec4	mulVec4R(const vec4& v) const;
	//multipy matrix by a vector from the left side
	__device__	vec4	mulVec4L(const vec4& v) const;
	//multipy matrix by a vector from the left AND right side
	__device__	float	mulVec4BothSides(const vec4& vl, const vec4& vr) const;

	//multipy matrix by a matrix from the right side
	__device__	matrix4	mulMat4R(const matrix4& mr) const;
	//multipy matrix by a matrix from the left side
	__device__	matrix4	mulMat4L(const matrix4& ml) const;
	//multipy matrix by a matrix from the right and left side
	__device__	matrix4	mulMat4BothSides(const matrix4& ml, const matrix4& mr) const;

	__device__	matrix4	transpose() const;



	float				_matrix[16];
};

__device__ matrix4::matrix4()
{
	for (int i = 0; i < 16; i++)
	{
		_matrix[i] = 0.0f;
	}
}

__device__ matrix4::matrix4(const vec4& col1, const vec4& col2, const vec4& col3, const vec4& col4)
{
	_matrix[0] = col1.x;
	_matrix[1] = col1.y;
	_matrix[2] = col1.z;
	_matrix[3] = col1.w;

	_matrix[4] = col2.x;
	_matrix[5] = col2.y;
	_matrix[6] = col2.z;
	_matrix[7] = col2.w;

	_matrix[8] = col3.x;
	_matrix[9] = col3.y;
	_matrix[10] = col3.z;
	_matrix[11] = col3.w;

	_matrix[12] = col4.x;
	_matrix[13] = col4.y;
	_matrix[14] = col4.z;
	_matrix[15] = col4.w;
}


__device__	matrix4 matrix4::transpose() const
{
	matrix4 result;
	int indexR = 0;
	for (int i = 0; i < 16; i++)
	{
		result._matrix[i] = _matrix[indexR];
		indexR += 4;
		if (indexR >= 16)
		{
			indexR -= 15;
		}
	}

	return result;
}

//multipy Matrix with another Matrix4 from the right side
__device__	matrix4 matrix4::mulMat4R(const matrix4& mr) const
{
	matrix4 result;

	for (int offset = 0; offset < 16; offset = offset + 4)
	{
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				result._matrix[i + offset] += mr._matrix[offset + j] * _matrix[i + 4 * j];
			}
		}
	}

	return result;
}

//multipy Matrix with another Matrix4 from the left side
__device__	matrix4 matrix4::mulMat4L(const matrix4& ml) const
{
	matrix4 result;

	for (int offset = 0; offset < 16; offset = offset + 4)
	{
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				result._matrix[i + offset] += ml._matrix[i + 4 * j] * _matrix[offset + j];
			}
		}
	}

	return result;
}

__device__ vec4 matrix4::getTranslation() const
{
	return vec4(_matrix[12], _matrix[13], _matrix[14], _matrix[15]);
}

//multiply Matrix with two matrices from both sides
__device__	matrix4 matrix4::mulMat4BothSides(const matrix4& ml, const matrix4& mr) const
{
	matrix4 result = mulMat4R(mr);
	result = result.mulMat4L(ml);

	return result;
}

__device__ void	matrix4::setProjectionRow(float a, float b, float c, float d)
{
	_matrix[3] = a;
	_matrix[7] = b;
	_matrix[11] = c;
	_matrix[15] = d;
}

__device__ void	matrix4::setTranslation(const vec3& trans)
{
	_matrix[12] = trans.x;
	_matrix[13] = trans.y;
	_matrix[14] = trans.z;
}

__device__ void matrix4::setRotationX(float angle)
{
	setIdentity();

	_matrix[5] = cos(angle);
	_matrix[6] = sin(angle);
	_matrix[9] = -sin(angle);
	_matrix[10] = cos(angle);
}

__device__ void matrix4::setRotationY(float angle)
{
	setIdentity();

	_matrix[0] = cos(angle);
	_matrix[2] = -sin(angle);
	_matrix[8] = sin(angle);
	_matrix[10] = cos(angle);
}

__device__ void matrix4::setRotationZ(float angle)
{
	setIdentity();

	_matrix[0] = cos(angle);
	_matrix[1] = sin(angle);
	_matrix[4] = -sin(angle);
	_matrix[5] = cos(angle);
}

//multipy matrix by a vector from the right side
__device__ vec4 matrix4::mulVec4R(const vec4& v) const
{
	vec4 result;
	result.x = _matrix[0] * v.x + _matrix[4] * v.y + _matrix[8] * v.z + _matrix[12] * v.w;
	result.y = _matrix[1] * v.x + _matrix[5] * v.y + _matrix[9] * v.z + _matrix[13] * v.w;
	result.z = _matrix[2] * v.x + _matrix[6] * v.y + _matrix[10] * v.z + _matrix[14] * v.w;
	result.w = _matrix[3] * v.x + _matrix[7] * v.y + _matrix[11] * v.z + _matrix[15] * v.w;
	return result;
}

//multipy matrix by a vector from the left side
__device__ vec4 matrix4::mulVec4L(const vec4& v) const
{
	vec4 result;
	result.x = _matrix[0] * v.x + _matrix[1] * v.y + _matrix[2] * v.z + _matrix[3] * v.w;
	result.y = _matrix[4] * v.x + _matrix[5] * v.y + _matrix[6] * v.z + _matrix[7] * v.w;
	result.z = _matrix[8] * v.x + _matrix[9] * v.y + _matrix[10] * v.z + _matrix[11] * v.w;
	result.w = _matrix[12] * v.x + _matrix[13] * v.y + _matrix[14] * v.z + _matrix[15] * v.w;
	return result;
}

__device__ float matrix4::mulVec4BothSides(const vec4& vl, const vec4& vr) const
{
	vec4 right = mulVec4R(vr);
	return right.x * vl.x + right.y * vl.y + right.z * vl.z + right.w * vl.w;
}


__device__ void	matrix4::setIdentity()
{
	_matrix[0] = 1.0f;
	_matrix[1] = 0.0f;
	_matrix[2] = 0.0f;
	_matrix[3] = 0.0f;

	_matrix[4] = 0.0f;
	_matrix[5] = 1.0f;
	_matrix[6] = 0.0f;
	_matrix[7] = 0.0f;

	_matrix[8] = 0.0f;
	_matrix[9] = 0.0f;
	_matrix[10] = 1.0f;
	_matrix[11] = 0.0f;

	_matrix[12] = 0.0f;
	_matrix[13] = 0.0f;
	_matrix[14] = 0.0f;
	_matrix[15] = 1.0f;
}


#endif