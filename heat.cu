#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#define CUDA_INPUT "input.txt"
#define CUDA_OUTPUT "cuda_output.txt"

int NUMPOINTS;
double ENDTIME;
double DT;
double ENDVALUES;

void InitialiseToZero(float* array);
__device__ void PrintPointsGPU(float* array, int size, double currentTime);
void PrintPointsCPU(float* array, double currentTime);
void ProcessOutput(float* array, int testCase, float time);
void CheckPoints(float* firstArray, float* secondArray);

__global__ void DiffuseHeat(float* currentPoints, float* nextPoints, const size_t size, double dx, double dt, const size_t endTime)
{
	unsigned int threadIndex = (threadIdx.x + blockDim.x * blockIdx.x) + 1;
 	double currentTime = 0.0;
	if (threadIndex > 0 && threadIndex < size-1)
	{	
		while (currentTime < endTime)
		{
			nextPoints[threadIndex] = currentPoints[threadIndex] + 0.25*(currentPoints[threadIndex+1] - (2*currentPoints[threadIndex]) + currentPoints[threadIndex-1]);
			__syncthreads();
			currentPoints[threadIndex] = nextPoints[threadIndex];
			currentTime += dt;
			__syncthreads();
		}
	}
}

void DiffuseHeatCPU(float* currentPoints, float* nextPoints, double dx, double dt, double endTime)
{
	double currentTime = 0.0;
	int index;
	while (currentTime < endTime)
	{
		for (index = 1; index < NUMPOINTS-1; index++)
		{
			nextPoints[index] = currentPoints[index] + 0.25*(currentPoints[index+1] - (2*currentPoints[index]) + currentPoints[index-1]);
		}
		for (index = 1; index < NUMPOINTS-1; index++)
		{
			currentPoints[index] = nextPoints[index];
		}
		currentTime += dt;
	}
}

int main(void)
{
	remove(CUDA_OUTPUT); //Deletes the old file since we only want output for current inputs
	int testCase = 1;
	char str[70];
	FILE *p;
	if((p=fopen(CUDA_INPUT,"r"))==NULL){
		printf("!!!Unable to open file cuda_input.txt!!!");
		exit(1);
	}
	while(fgets(str,70,p)!=NULL) //Read the file line by line until the end
	{
		//Get the required values from the input file
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

		const double endTime = ENDTIME;
		const int size = NUMPOINTS;

		//Initialise all the required arrays
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
		//Set the initial points to all be zero
		InitialiseToZero(currentPoints);
		InitialiseToZero(nextPoints);

		//Send the end values to the specified values
		currentPoints[0] = ENDVALUES;
		currentPoints[NUMPOINTS-1] = ENDVALUES;
		
		//Copy the arrays to the device
		cudaMemcpy(deviceCurrentPoints, currentPoints, NUMPOINTS*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(deviceNextPoints, nextPoints, NUMPOINTS*sizeof(float), cudaMemcpyHostToDevice);
		
		//Set up the blocks and grid
		const size_t blockSize = 256;
		size_t gridSize = (NUMPOINTS-2) / blockSize;
		if ((NUMPOINTS-2)%blockSize) gridSize++;
		double DX = currentPoints[1] - currentPoints[0];

		//Set up the timing and calculate the heat diffusion on the GPU
		cudaEvent_t launch_begin, launch_end;
		cudaEventCreate(&launch_begin);
		cudaEventCreate(&launch_end);
		cudaEventRecord(launch_begin,0);
		DiffuseHeat<<<gridSize, blockSize>>>(deviceCurrentPoints, deviceNextPoints, size, DX, DT, endTime);
		cudaEventRecord(launch_end,0);
		cudaEventSynchronize(launch_end);
		float timeCuda = 0;
		cudaEventElapsedTime(&time, launch_begin, launch_end);

		//Copy the result from the device to the host
		cudaMemcpy(resultPoints, deviceCurrentPoints, NUMPOINTS*sizeof(float), cudaMemcpyDeviceToHost);

		//Record the runtime and result in the output text file
		ProcessOutput(resultPoints, testCase, time);
		
		//Set up the timing for the serial version and calcuate the heat diffusion on the CPU
		clock_t start = clock(), diff;
		DiffuseHeatCPU(currentPoints, nextPoints, DX, DT, ENDTIME);
		diff = clock() - start;
		int timeSerial = diff * 1000 / CLOCKS_PER_SEC;
		
		//Print the results and the speedup
		printf("The serial runtime was %d\n", timeSerial);
		printf("The CUDA runtime was %0.1f\n", timeCuda);
		printf("This results in a speedup of %0.1f\n, timeSerial/timeCuda);
		CheckResults(currentPoints, resultPoints);
		
		testCase++;
		free(currentPoints);
		free(nextPoints);
		free(resultPoints);
		cudaFree(deviceCurrentPoints);
		cudaFree(deviceNextPoints);
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

void CheckPoints(float* firstArray, float* secondArray)
{
	int index;
	valid = 1;
	for (index = 0; index < NUMPOINTS; index++)
	{
		if (firstArray[index] != secondArray[index]) valid = 0;
	}
	if (valid == 0) printf("The two resultant temperature arrays are not the same");
	else printf("The two resultant temperature arrays are the same");
}

void ProcessOutput(float* array, int testCase, float time)
{
	/*FILE *f = fopen(CUDA_OUTPUT, "a");
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
	fclose(f);*/
	printf("Runtime for test case %d with %d points:\n", testCase, NUMPOINTS);
	printf("%f\n", time);
	if (NUMPOINTS <= 15) {
		printf("Resultant temperatures:\n");
		for (int index = 0; index < NUMPOINTS; index++)
		{
			printf("%0.2f ", array[index]);
		}
	}
	printf("\n\n");
}
