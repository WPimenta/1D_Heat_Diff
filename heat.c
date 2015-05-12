#include <stdlib.h>
#include <stdio.h>

#define NUMPOINTS 10
#define ENDTIME 10
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
	for (int index = 0; index < NUMPOINTS; index++)
	{
		currentPoints[index] = rand()%100;
	}
	
	double DX = currentPoints[1] - currentPoints[0];
	DiffuseHeat(currentPoints, nextPoints, NUMPOINTS, DX, DT, ENDTIME);
	return 1;
}

void InitialiseToZero(float* array, int size)
{
	for (int index = 0; index < size; index++)
	{
		array[index] = 0;
	}
}

void DiffuseHeat(float* currentPoints, float* nextPoints, int size, double dx, double dt, double endTime)
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

