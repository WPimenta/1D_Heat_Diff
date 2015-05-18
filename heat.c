#include <stdlib.h>
#include <stdio.h>

#define NUMPOINTS 10
#define ENDTIME 5
#define DT 0.1

void InitialiseToZero(float* array, int size);
void DiffuseHeat(float* currentPoints, float* nextPoints, int size, double dx, double dt, double endTime);
void PrintPoints(float* array, int size, double currentTime);

int main(void)
{
	float* currentPoints = 0;
	float* nextPoints = 0;
	currentPoints = (float*)malloc(NUMPOINTS*sizeof(float));
	nextPoints = (float*)malloc(NUMPOINTS*sizeof(float));

	InitialiseToZero(currentPoints, NUMPOINTS);
	InitialiseToZero(nextPoints, NUMPOINTS);

	//make the end points some random values
	double randomValue = rand()%99;
	printf("%f\n", randomValue);
	currentPoints[0] = randomValue;
	currentPoints[NUMPOINTS-1] = randomValue;
	
	double DX = currentPoints[1] - currentPoints[0];
	DiffuseHeat(currentPoints, nextPoints, NUMPOINTS, DX, DT, ENDTIME);
	return 1;
}

void InitialiseToZero(float* array, int size)
{
	int index;
	for (index = 0; index < size; index++)
	{
		array[index] = 0;
	}
}

void DiffuseHeat(float* currentPoints, float* nextPoints, int size, double dx, double dt, double endTime)
{
	double currentTime = 0.0;
	int index;
	while (currentTime < endTime)
	{
		for (index = 1; index < size-1; index++)
		{
			nextPoints[index] = currentPoints[index] + 0.25*(currentPoints[index+1] - (2*currentPoints[index]) + currentPoints[index-1]);
		}
		for (index = 1; index < size-1; index++)
		{
			currentPoints[index] = nextPoints[index];
		}
		currentTime += dt;
		PrintPoints(currentPoints, size, currentTime);
	}
	PrintPoints(currentPoints, size, currentTime);
}

void PrintPoints(float* array, int size, double currentTime)
{
	printf("The array values at time t=%0.1f are:\n", currentTime);
	int index;
	for (index = 0; index < size; index++)
	{
		printf("%0.2f ", array[index]);
	}
	printf("\n\n");
}

