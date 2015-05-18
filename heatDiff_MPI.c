//Header files inclided and definitions.
//
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <stdio.h>
#define NUMPOINTS 10
#define ENDTIME 1
#define DT 0.1
#define NUM_ELEMENTS 3
#define ROOM_TEMP 0

//Function declarations.
//
void InitRod(float* array, int size, float roomTemp, double appliedHeat);
void DiffuseHeat(float* in, float* currentPoints, int size, double apliedHeat, double dx, double dt);
void PrintPoints(float* array, int size, double currentTime);

int main()
{
	double currentTime = 0.0;
	int n;
	int chunk_size;
	float* currentPoints = NULL;
	float appliedHeat;
	int w_rank;
	int w_size;
	double dx;
	double dt;
	double timing;
	MPI_Status status;

	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &w_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &w_size);

	if(w_rank == 0)
	{
		srand(time(NULL));
		currentPoints = (float*)malloc(NUMPOINTS*sizeof(float));
		appliedHeat = rand()%100;
		printf("%f is applied heat\n",appliedHeat);
		InitRod(currentPoints, NUMPOINTS, ROOM_TEMP, appliedHeat);
		dx = currentPoints[1] - currentPoints[0];
		n = NUMPOINTS;
		dt = DT;
		chunk_size = n/w_size;
	}

	MPI_Barrier(MPI_COMM_WORLD);
	timing -= MPI_Wtime();

	MPI_Bcast(&appliedHeat, 1, MPI_FLOAT, 0 , MPI_COMM_WORLD);
	MPI_Bcast(&n, 1, MPI_INT, 0 , MPI_COMM_WORLD);
	MPI_Bcast(&dx, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&dt, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&currentPoints, NUMPOINTS, MPI_FLOAT, 0, MPI_COMM_WORLD);
	float *process_recv_buffer = (float*)malloc(chunk_size*sizeof(float));
	while(currentTime < ENDTIME)
	{
		MPI_Scatter(currentPoints, chunk_size, MPI_FLOAT, process_recv_buffer, chunk_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
		DiffuseHeat(process_recv_buffer, currentPoints, chunk_size, appliedHeat,dx, dt);
		currentTime += dt;
		printf("%f is dt\n", currentTime);
	}
	free(currentPoints); free(process_recv_buffer);
	MPI_Finalize();
	return 1;
}


void InitRod(float* array, int size, float roomTemp, double appliedHeat)
{
	int index;
	for (index = 0; index < size; index++)
	{
		array[index] = roomTemp;
	}
	array[0] = appliedHeat;
	array[NUMPOINTS-1] = appliedHeat;
}


void DiffuseHeat(float* in, float* currentPoints, int size, double appliedHeat,double dx, double dt)
{
	//create temporary storage array
	int w_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &w_rank);
	int index;
	printf("%f is the first element of CurrentPOints from %d\n", currentPoints[0], w_rank);
	for(index = 1; index < size-1; index++)
	{
		currentPoints[index] = currentPoints[index] + (dt/dx*dx)*(currentPoints[index+1] - (2*currentPoints[index]) + currentPoints[index-1]);
	}
	//call merge
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
