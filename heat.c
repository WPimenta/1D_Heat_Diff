#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <sys/time.h>
#define SERIAL_INPUT "input.txt"
#define SERIAL_OUTPUT "serial_output.txt"

int NUMPOINTS;
double ENDTIME;
double DT;
double ENDVALUES;

void InitialiseToZero(float* array);
void DiffuseHeat(float* currentPoints, float* nextPoints, double dx, double dt, double endTime);
void PrintPoints(float* array, double currentTime);
void ProcessOutput(float* array, int testCase, float time);

int main(void)
{
	remove(SERIAL_OUTPUT);
	int testCase = 1;
	char str[70];
	FILE *p;
	
	if((p=fopen(SERIAL_INPUT,"r"))==NULL){
		printf("!!!Unable to open file input.txt!!!");
		exit(1);
	}
	while(fgets(str,70,p)!=NULL)
	{
		//File is in format: NUMPOINTS ENDTIME DT ENDVALUES
		char * pch;
		pch = strtok(str," ");
		NUMPOINTS = atof(pch);
		pch = strtok(NULL, " ");
		ENDTIME = atof(pch);
		pch = strtok(NULL, " ");
		DT = atof(pch);
		pch = strtok(NULL, " ");
		ENDVALUES = atof(pch);

		float* currentPoints = 0;
		currentPoints = (float*)malloc(NUMPOINTS*sizeof(float));
		float* nextPoints = 0;	
		nextPoints = (float*)malloc(NUMPOINTS*sizeof(float));

		InitialiseToZero(currentPoints);
		InitialiseToZero(nextPoints);

		currentPoints[0] = ENDVALUES;
		currentPoints[NUMPOINTS-1] = ENDVALUES;
	
		double DX = currentPoints[1] - currentPoints[0];

		clock_t start = clock(), diff;
		DiffuseHeat(currentPoints, nextPoints, DX, DT, ENDTIME);
		diff = clock() - start;
		double time = diff * 1000 / CLOCKS_PER_SEC;

		ProcessOutput(currentPoints, testCase, time);
		testCase++;
		free(currentPoints);
		free(nextPoints);
	}
	return 1;
}

void InitialiseToZero(float* array)
{
	int index;
	for (index = 0; index < NUMPOINTS; index++)
	{
		array[index] = 0;
	}
}

void DiffuseHeat(float* currentPoints, float* nextPoints, double dx, double dt, double endTime)
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

void PrintPoints(float* array, double currentTime)
{
	printf("The array values at time t=%0.1f are:\n", currentTime);
	int index;
	for (index = 0; index < NUMPOINTS; index++)
	{
		printf("%0.2f ", array[index]);
	}
	printf("\n\n");
}

void ProcessOutput(float* array, int testCase, float time)
{
	/*//int index;
	FILE *f = fopen(SERIAL_OUTPUT, "a");
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
	fprintf(f, "%f\n", time);
	if (NUMPOINTS <= 15) {
		printf("Resultant temperatures:\n");
		for (int index = 0; index < NUMPOINTS; index++)
		{
			printf("%0.2f ", array[index]);
		}
	}
	printf("\n\n");
}
