#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define NUMPOINTS 10
#define ENDTIME 10
#define DT 0.1

void InitialiseToZero(float* array, int size);
void DiffuseHeatCPU(float* currentPoints, float* nextPoints, int size, double dx, double dt, double endTime);
void PrintPoints(float* array, int size, double currentTime);

__global__ void DiffuseHeat(float* currentPoints, float* nextPoints, int size, double dx, double dt, double endTime)
{
	unsigned int index = (threadIdx.x + blockDim.x * blockIdx.x) + 1;
	__shared__ double currentTime;
	currentTime = 0.0;
	while (currentTime < endTime)
	{	
		nextPoints[index] = currentPoints[index] + (dt/dx*dx)*(currentPoints[index+1] - (2*currentPoints[index]) + currentPoints[index-1]);				
		__syncthreads();
		currentPoints[index] = nextPoints[index];
		if (index == 1)	
		{
			printf("The array values at time t=%0.1f are:\n", currentTime);
			for (int index = 0; index < size; index++)
			{
				printf("%0.2f ", currentPoints[index]);
			}
			printf("\n\n");
			currentTime += dt;
		}
		__syncthreads();
	}
}

int main(void)
{
	float* currentPoints = 0;
	currentPoints = (float*)malloc(NUMPOINTS*sizeof(float));
	float* nextPoints = 0;
	nextPoints = (float*)malloc(NUMPOINTS*sizeof(float));
	float* resultPoints = 0;
	resultPoints = (float*)malloc(NUMPOINTS*sizeof(float));
	
	float* deviceCurrentPoints = 0;
	cudaMalloc((void**)&deviceCurrentPoints, NUMPOINTS*sizeof(float));
	float* deviceNextPoints = 0;
	cudaMalloc((void**)&deviceNextPoints, NUMPOINTS*sizeof(float));

	if(currentPoints == 0 || nextPoints == 0 || resultPoints == 0 || deviceCurrentPoints == 0 || deviceNextPoints == 0)
	{
		printf("Couldn't allocate memory\n");
		return 1;
	}

	InitialiseToZero(currentPoints, NUMPOINTS);
	InitialiseToZero(nextPoints, NUMPOINTS);

	//make the end points some random values
	float randomValue = rand()%100;
	currentPoints[0] = randomValue;
	currentPoints[NUMPOINTS-1] = randomValue;
	
	cudaMemcpy(deviceCurrentPoints, currentPoints, NUMPOINTS*sizeof(float), cudaMemcpyHostToDevice);
  	cudaMemcpy(deviceNextPoints, nextPoints, NUMPOINTS*sizeof(float), cudaMemcpyHostToDevice);

	const size_t blockSize = NUMPOINTS-2;
  	size_t gridSize = (NUMPOINTS-2) / blockSize;
	double DX = currentPoints[1] - currentPoints[0];
	DiffuseHeat<<<gridSize, blockSize>>>(deviceCurrentPoints, deviceNextPoints, NUMPOINTS, DX, DT, ENDTIME);
	cudaMemcpy(resultPoints, deviceCurrentPoints, NUMPOINTS*sizeof(float), cudaMemcpyDeviceToHost);
	PrintPoints(resultPoints, NUMPOINTS, ENDTIME);
	return 0;
}

void InitialiseToZero(float* array, int size)
{
	for (int index = 0; index < size; index++)
	{
		array[index] = 0;
	}
}

void DiffuseHeatCPU(float* currentPoints, float* nextPoints, int size, double dx, double dt, double endTime)
{
	float initialStart = currentPoints[0];
	float initialEnd = currentPoints[size-1];
	double currentTime = 0.0;
	while (currentTime < endTime)
	{
		PrintPoints(currentPoints, size, currentTime);
		for (int index = 1; index < size-1; index++)
		{
			nextPoints[index] = currentPoints[index] + (dt/dx*dx)*(currentPoints[index+1] - (2*currentPoints[index]) + currentPoints[index-1]);
		}
		currentPoints = nextPoints;
		currentPoints[0] = initialStart;
		currentPoints[size-1] = initialEnd;
		currentTime += dt;
	}
	PrintPoints(currentPoints, size, currentTime);
}

void PrintPoints(float* array, int size, double currentTime)
{
	printf("The array values at time t=%0.1f are:\n", currentTime);
	for (int index = 0; index < size; index++)
	{
		printf("%0.2f ", array[index]);
	}
	printf("\n\n");
}
