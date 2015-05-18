#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#define CUDA_INPUT "input.txt"
#define CUDA_OUTPUT "output.txt"

int NUMPOINTS;
double ENDTIME;
double DT;
double ENDVALUES;

void InitialiseToZero(float* array);
__device__ void PrintPointsGPU(float* array, int size, double currentTime);
void PrintPointsCPU(float* array, double currentTime);
void ProcessOutput(float* array, int testCase, float time);

__global__ void DiffuseHeat(float* currentPoints, float* nextPoints, int size, double dx, double dt, double endTime)
{
	unsigned int threadIndex = (threadIdx.x + blockDim.x * blockIdx.x) + 1;
	__shared__ double currentTime;
	currentTime = 0.0;
	while (currentTime < endTime)
	{
		nextPoints[threadIndex] = currentPoints[threadIndex] + 0.25*(currentPoints[threadIndex+1] - (2*currentPoints[threadIndex]) + currentPoints[threadIndex-1]);
		__syncthreads();
		currentPoints[threadIndex] = nextPoints[threadIndex];
		if (threadIndex == 1)
		{
			currentTime += dt;
		}
		__syncthreads();
	}
}
int main(void)
{
	remove(CUDA_OUTPUT);
	int testCase = 1;
	char str[70];
	FILE *p;
	if((p=fopen(CUDA_INPUT,"r"))==NULL){
		printf("!!!Unable to open file cuda_input.txt!!!");
		exit(1);
	}
	while(fgets(str,70,p)!=NULL)
	{
		//File is in format: NUMPOINTS ENDTIME DT ENDVALUES
		char * pch;
		pch = strtok (str," ");
		NUMPOINTS = atof(pch);
		pch = strtok (NULL, " ");
		ENDTIME = atof(pch);
		pch = strtok (NULL, " ");
		DT = atof(pch);
		pch = strtok (NULL, " ");
		ENDVALUES = atof(pch);

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
		InitialiseToZero(currentPoints);
		InitialiseToZero(nextPoints);

		currentPoints[0] = ENDVALUES;
		currentPoints[NUMPOINTS-1] = ENDVALUES;
		cudaMemcpy(deviceCurrentPoints, currentPoints, NUMPOINTS*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(deviceNextPoints, nextPoints, NUMPOINTS*sizeof(float), cudaMemcpyHostToDevice);
		const size_t blockSize = NUMPOINTS-2;
		size_t gridSize = (NUMPOINTS-2) / blockSize;
		double DX = currentPoints[1] - currentPoints[0];

		cudaEvent_t launch_begin, launch_end;
		cudaEventCreate(&launch_begin);
		cudaEventCreate(&launch_end);
		cudaEventRecord(launch_begin,0);
		DiffuseHeat<<<gridSize, blockSize>>>(deviceCurrentPoints, deviceNextPoints, NUMPOINTS, DX, DT, ENDTIME);
		cudaEventRecord(launch_end,0);
		cudaEventSynchronize(launch_end);
		float time = 0;
		cudaEventElapsedTime(&time, launch_begin, launch_end);

		cudaMemcpy(resultPoints, deviceCurrentPoints, NUMPOINTS*sizeof(float), cudaMemcpyDeviceToHost);

		ProcessOutput(resultPoints, testCase, time);
		testCase++;
	}
	fclose(p);
	return 0;
}
void InitialiseToZero(float* array)
{
	for (int index = 0; index < NUMPOINTS; index++)
	{
		array[index] = 0;
	}
}

__device__ void PrintPointsGPU(float* array, int size, double currentTime)
{
	printf("The array values at time t=%0.1f are:\n", currentTime);
	for (int index = 0; index < size; index++)
	{
		printf("%0.2f ", array[index]);
	}
	printf("\n\n");
}

void PrintPointsCPU(float* array, double currentTime)
{
	printf("The array values at time t=%0.1f are:\n", currentTime);
	for (int index = 0; index < NUMPOINTS; index++)
	{
		printf("%0.2f ", array[index]);
	}
	printf("\n\n");
}

void ProcessOutput(float* array, int testCase, float time)
{
	FILE *f = fopen(CUDA_OUTPUT, "a");
	if (f == NULL)
	{
		printf("!!!Error opening the output file!!!\n");
		exit(1);
	}
	fprintf(f, "Runtime for test case %d with %d points:\n", testCase, NUMPOINTS);
	fprintf(f, "%f\n", time);
	if (NUMPOINTS <= 15) {
		fprintf(f, "Resultant temperatures:\n");
		for (int index = 0; index < NUMPOINTS; index++)
		{
			fprintf(f, "%0.2f ", array[index]);
		}
	}
	fprintf(f, "\n\n");
	fclose(f);
}
