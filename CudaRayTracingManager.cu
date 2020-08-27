#ifndef CUDA_RAY_TRACING_MANAGER_CU
#define CUDA_RAY_TRACING_MANAGER_CU

#include "CudaRayTracingManager.cuh"
#include "Constants.cuh"

#include "Ray.cu"
#include "Hit.cu"
#include "RigidEntityList.cu"
#include "MaterialList.cu"
#include "SeedGenerator.cu"
#include "LightList.cu"
#include "Camera.cu"
#include "Utils.h"


__device__ CudaRayTracingManager::CudaRayTracingManager(CudaParams& params)
: _imageWidth(params.imgw), _imageHeight(params.imgh), _frames(0)
{
	_materials = new MaterialList();
	_entities = new RigidEntityList(_materials);
	_seedGen = new SeedGenerator(params.countRandomNumbers);
	_lights = new LightList();

	vec4 eyePos(0, 0, 30, 1);
	_camera = new Camera(eyePos);

	_useAntialiasing = false;
}

__device__ CudaRayTracingManager::~CudaRayTracingManager()
{	
	delete _camera;
	delete _materials;
	delete _entities;
	delete _seedGen;
}

__device__ void CudaRayTracingManager::prepareForNextFrame(CudaParams& params)
{
	_seedGen->incrementSeeds();
}

__device__ void	CudaRayTracingManager::concludeFrame(CudaParams& params)
{
	_frames += 1;

	if (params.currentMouseButton == 0)
	{
		if (params.mouseState == 0)
		{
			_camera->setAction(CameraAction::rotate, params.mousePositionX, params.mousePositionY);
			_camera->act(params.mousePositionX, params.mousePositionY);
			_frames = 0;
		}
		else
		{
			_camera->setAction(CameraAction::none);
			return;
		}
	}
	else if (params.currentMouseButton == 3 || params.currentMouseButton == 4)
	{
		if (params.mouseState == GLUT_UP) return;
		if (params.currentMouseButton == 3)
		{
			_camera->setAction(CameraAction::scrollUp, params.mousePositionX, params.mousePositionY);
		}
		else
		{
			_camera->setAction(CameraAction::scrollDown, params.mousePositionX, params.mousePositionY);
		}
		_camera->act(params.mousePositionX, params.mousePositionY);
		_frames = 0;
		_camera->setAction(CameraAction::none);
	}
}


__device__ vec3 CudaRayTracingManager::processRayTracing(CudaParams& params, curandState_t& randomState)
{
	if (_useAntialiasing == true)
	{
		_useAntialiasing = false;
		_frames = 0;
	}

	int index = blockIdx.x*blockDim.x + threadIdx.x;
	vec3 directLight(0, 0, 0);
	vec3 indirectLight(0,0,0);
	Ray primaryRay;
	Ray* primArray = *((Ray**)params.primaryRays);
	Hit firstHit;

	if (_frames == 0)
	{
		clearBuffer(params);
	}
	directLight = getDirectLighting(params, randomState, primaryRay, &primArray, firstHit);
	if (firstHit.getMaterial() == NULL) return directLight;

	Ray scatterRay = getScatterRay(primaryRay, firstHit, randomState);
	indirectLight = getIndirectLight(scatterRay, firstHit);
	sumUpIndirectLight(indirectLight, params);

	//Necessary for faster camera movement
	if (_frames == 0)
	{
		return directLight;
	}
	
	return directLight.add(indirectLight);
}



__device__ vec3 CudaRayTracingManager::processRayTracingWithAntialiasing(CudaParams& params, curandState_t& randomState)
{
	if (_useAntialiasing == false)
	{
		_useAntialiasing = true;
		_frames = 0;
	}

	int index = blockIdx.x*blockDim.x + threadIdx.x;
	vec3 directLight(0, 0, 0);
	vec3 indirectLight(0, 0, 0);
	Ray primaryRay;
	Ray* primArray = *((Ray**)params.primaryRays);
	Hit firstHit;

	if (_frames == 0)
	{
		clearBuffer(params);
	}

	int samples = 2;

	for (int i = 0; i < samples; i++)
	{
		primaryRay = Ray(_camera->getEyePosition(), _camera->getRayDirection(params.imgw, params.imgh, randomState));

		if (primaryRay.trace(*_entities) == false)
		{
			directLight = directLight.add(vec3(AMBIENT_COLOR_R, AMBIENT_COLOR_G, AMBIENT_COLOR_B));
			continue;
		}

		firstHit = primaryRay.getHit();
		vec4 reflectionDir = Utils::getReflectionVector(firstHit.getHitPosition().getNormal(), primaryRay.getDirection());
		directLight = directLight.add(getLightAtPoint(firstHit.getHitPosition(), firstHit.getMaterial(), _camera->getEyePosition(), reflectionDir));
	
		
		Ray scatterRay = getScatterRay(primaryRay, firstHit, randomState);
		vec3 newIndirectLight(0, 0, 0);
		
		if (scatterRay.trace(*_entities))
		{
			Hit secondHit = scatterRay.getHit();

			vec4 reflectionDir = Utils::getReflectionVector(secondHit.getHitPosition().getNormal(), scatterRay.getDirection());
			newIndirectLight = getLightAtPoint(secondHit.getHitPosition(), secondHit.getMaterial(), firstHit.getHitPosition().getPosition(), reflectionDir);

			newIndirectLight = newIndirectLight.mul(firstHit.getMaterial()->getColor());
			if (firstHit.getMaterial()->isReflective() == false)
			{
				float scatterRayCosine = firstHit.getHitPosition().getNormal().dot(scatterRay.getDirection());
				newIndirectLight = newIndirectLight.mul(scatterRayCosine);
			}
		}
		else
		{
			newIndirectLight = vec3(AMBIENT_COLOR_R, AMBIENT_COLOR_G, AMBIENT_COLOR_B).mul(AMBIENT_LIGHT_INDIRECT_INTENSITY);
			newIndirectLight = newIndirectLight.mul(firstHit.getMaterial()->getColor());
		}
		indirectLight = indirectLight.add(newIndirectLight);
	}
	directLight = directLight.div(samples);

	if (_frames == 0)
	{
		params.directLighting[index] = Utils::rgbToInt(directLight.x * 255, directLight.y * 255, directLight.z * 255);
	}
	else
	{
		vec3 formerDirectLight = Utils::intToRGB(params.directLighting[index]);
		directLight = directLight.add(formerDirectLight.div(255).mul(_frames)).div(_frames + 1);
	}

	indirectLight = indirectLight.div(samples);
	sumUpIndirectLight(indirectLight, params);
	
	//Necessary for faster camera movement
	if (_frames == 0)
	{
		return directLight;
	}

	indirectLight = Utils::intToRGB(params.indirectLighting[index]).div(255);

	return directLight.add(indirectLight);
}


__device__ vec3 CudaRayTracingManager::getLightAtPoint(HitPosition& hpos, const Material* mat, vec4 eyePos, vec4 reflectionDir)
{
	vec3 color(0, 0, 0);
	int countDirectionalLights = _lights->getNumDirectionalLights();
	vec4 position = hpos.getPosition();
	vec4 normal = hpos.getNormal();
	vec3 matColor = mat->getColor();
	vec4 viewDir = eyePos.sub(position);
	viewDir = viewDir.normalize();

	Ray shadowRay;
	
	int countEntities = _entities->getNumEntities();

	//calculate directional light influence
	for (int i = 0; i < countDirectionalLights; i++)
	{
		DirectionalLight* dlight = _lights->getDirectionalLights()[i];

		//the direction from a DirectionalLight is considered from the "suns" direction. 
		//For that reason the direction is inverted
		vec4 lightDirection = vec4(dlight->getDirection(), 0).mul(-1);
		shadowRay.setDirection(lightDirection);
		shadowRay.setStart(position.add(shadowRay.getDirection().mul(0.0001f)));

		//check if the back of the surface is facing towards the light source
		if (lightDirection.dot(normal) < 0.0f) continue;
		if (shadowRay.trace(*_entities)) continue;

		float cosineFactor = max(normal.dot(lightDirection), 0.0f);
		vec3 diffuseBRDF = mat->getBRDF()->getDiffuseBRDF(matColor);
		float specularBRDF = mat->getBRDF()->getSpecularBRDF(reflectionDir, viewDir);

		vec3 lightColor = dlight->getColor().mul(dlight->getIntensity() * cosineFactor);

		vec3 diffuseColor = diffuseBRDF.mul(lightColor);
		vec3 specularColor = lightColor.mul(specularBRDF);

		color = color.add(diffuseColor);
		color = color.add(specularColor);
	}

	return color;
}

__device__ Ray CudaRayTracingManager::getScatterRay(Ray& primaryRay, Hit& firstHit, curandState_t& randomState)
{
	vec4 newRayDirection;
	if (firstHit.getMaterial()->isReflective())
	{
		newRayDirection = Utils::getReflectionVector(firstHit.getHitPosition().getNormal(), primaryRay.getDirection());
	}
	else
	{
		newRayDirection = getPointOnHemisphere(firstHit, randomState).normalize();
	}
	vec4 startPos = firstHit.getHitPosition().getPosition();
	startPos = startPos.add(newRayDirection.mul(0.0001f));
	return Ray(startPos, newRayDirection);
}

__device__ vec3 CudaRayTracingManager::getIndirectLight(Ray& scatterRay, Hit& firstHit)
{
	vec3 indirectLight(0, 0, 0);

	if (scatterRay.trace(*_entities))
	{
		Hit secondHit = scatterRay.getHit();

		vec4 reflectionDir = Utils::getReflectionVector(secondHit.getHitPosition().getNormal(), scatterRay.getDirection());
		indirectLight = getLightAtPoint(secondHit.getHitPosition(), secondHit.getMaterial(), firstHit.getHitPosition().getPosition(), reflectionDir);

		indirectLight = indirectLight.mul(firstHit.getMaterial()->getColor());
		if (firstHit.getMaterial()->isReflective() == false)
		{
			float scatterRayCosine = firstHit.getHitPosition().getNormal().dot(scatterRay.getDirection());
			indirectLight = indirectLight.mul(scatterRayCosine);
		}
	}
	else
	{
		indirectLight = vec3(AMBIENT_COLOR_R, AMBIENT_COLOR_G, AMBIENT_COLOR_B).mul(firstHit.getMaterial()->getColor());
		indirectLight = indirectLight.mul(AMBIENT_LIGHT_INDIRECT_INTENSITY);
	}

	return indirectLight;
}


//Creates a ray in the center of the pixel
__device__ Ray CudaRayTracingManager::createRay(CudaParams& params, curandState_t& randomState)
{
	vec4 dir = _camera->getRayDirection(params.imgw, params.imgh);
	return Ray(_camera->getEyePosition(), dir);
}

//formula for hemisphere point from "Ray Tracing from the Ground Up - p.128"
__device__ vec4 CudaRayTracingManager::getPointOnHemisphere(Hit& hit, curandState_t& randomState)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	vec4 normal = hit.getHitPosition().getNormal();
	vec4 position = hit.getHitPosition().getPosition();

	//create orthonormal basis, based on "Ray Tracing from the Ground Up - p.311"
	vec4 w = normal;
	vec4 up(0,1,0,0);
	if (w.x <= 0.001f && w.y > 0 && w.z <= 0.001f)
	{
		up.y = 0.5f;
		up.x = 0.5f;
		up = up.normalize();
	}
	vec4 v = w.cross(up);
	v = v.normalize();
	vec4 u = v.cross(w);

	float r1 = curand_uniform(&randomState);
	float r2 = curand_uniform(&randomState);

	float e = 5.0f;
	float exponent = 1.0f / (e + 1.0f);
	float theta = acos(powf(1-r2, exponent));
	float phi = 2 * M_PI * r1;

	float sineTheta = sin(theta);
	float cosineTheta = cos(theta);
	float sinePhi = sin(phi);
	float cosinePhi = cos(phi);
	float a = sineTheta * cosinePhi;
	float b = sineTheta * sinePhi;

	u = u.mul(sineTheta * cosinePhi);
	v = v.mul(sineTheta * sinePhi);
	w = w.mul(cosineTheta);

	vec4 hemisP = u.add(v).add(w);

	return hemisP;
}

__device__ void CudaRayTracingManager::sumUpIndirectLight(vec3& indirectLight, CudaParams& params)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	
	indirectLight = indirectLight.mul(255);
	if (_frames == 0)
	{
		params.indirectLighting[index] = Utils::rgbToInt(indirectLight.x, indirectLight.y, indirectLight.z);
	}
	else {
		vec3 color = Utils::intToRGB(params.indirectLighting[index]);
		color = color.mul(_frames);
		color = color.add(indirectLight);
		indirectLight = color.div(_frames + 1);
		params.indirectLighting[index] = Utils::rgbToInt(indirectLight.x, indirectLight.y, indirectLight.z);
	}
	indirectLight = indirectLight.div(255);
}

__device__ vec3 CudaRayTracingManager::getDirectLighting(CudaParams& params, curandState_t& randomState, Ray& primaryRay, Ray** primArray, Hit& firstHit)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	vec3 directLight(0,0,0);
	if (_frames == 0)
	{
		primaryRay = createRay(params, randomState);

		if (primaryRay.trace(*_entities) == false)
		{
			params.directLighting[index] = Utils::rgbToInt(AMBIENT_COLOR_R * 255, AMBIENT_COLOR_G * 255, AMBIENT_COLOR_B * 255);
			(*primArray)[index] = primaryRay;
			return vec3(AMBIENT_COLOR_R, AMBIENT_COLOR_G, AMBIENT_COLOR_B);
		}
		(*primArray)[index] = primaryRay;

		firstHit = primaryRay.getHit();
		vec4 reflectionDir = Utils::getReflectionVector(firstHit.getHitPosition().getNormal(), primaryRay.getDirection());
		directLight = getLightAtPoint(firstHit.getHitPosition(), firstHit.getMaterial(), _camera->getEyePosition(), reflectionDir);

		params.directLighting[index] = Utils::rgbToInt(directLight.x * 255, directLight.y * 255, directLight.z * 255);
	}
	else
	{
		primaryRay = (*primArray)[index];
		firstHit = primaryRay.getHit();
		if (firstHit.getMaterial() == NULL) return vec3(AMBIENT_COLOR_R, AMBIENT_COLOR_G, AMBIENT_COLOR_B);

		directLight = Utils::intToRGB(params.directLighting[index]);
		directLight = directLight.div(255);
	}
	return directLight;
}

__device__ void CudaRayTracingManager::clearBuffer(CudaParams& params)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	params.directLighting[index] = 0;
	params.indirectLighting[index] = 0;
	params.g_odata[index] = 0;
}

__device__ SeedGenerator* CudaRayTracingManager::getSeedGenerator()
{
	return _seedGen;
}

__device__ int CudaRayTracingManager::getCountFrames()
{
	return _frames;
}


#endif