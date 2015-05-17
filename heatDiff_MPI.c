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
void DiffuseHeat(float* currentPoints, int size, double apliedHeat, float first, float last, double dx, double dt);
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
		float first;
		float last;
		srand(time(NULL));
		currentPoints = (float*)malloc(NUMPOINTS*sizeof(float));
		appliedHeat = rand()&100;
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

	float *process_recv_buffer = (float*)malloc(chunk_size*sizeof(float));
	float first;
	float last;
	while(currentTime < ENDTIME)
	{
		if(w_rank==0)
		{
			first = currentPoints[0];
			last = currentPoints[chunk_size];
		}
		if(w_rank-1 >= 1)
		{
			MPI_Send(&currentPoints[w_rank*chunk_size], 1, MPI_FLOAT, w_rank-1, w_rank, MPI_COMM_WORLD);
		}
		if(w_rank + 1 <= w_size)
		{
			MPI_Send(&currentPoints[w_rank*chunk_size-1+chunk_size], 1, MPI_FLOAT, w_rank+1, w_rank, MPI_COMM_WORLD);
		}
		if(w_rank-1 >= 1)
		{
			MPI_Recv(&first, 1, MPI_FLOAT,w_rank, w_rank-1, MPI_COMM_WORLD, &status );
		}
		if(w_rank + 1 <= w_size)
		{
			MPI_Recv(&last, 1, MPI_FLOAT, w_rank, w_rank+1, MPI_COMM_WORLD, &status);
		}

		MPI_Scatter(currentPoints, chunk_size, MPI_FLOAT, process_recv_buffer, chunk_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

		DiffuseHeat(process_recv_buffer, chunk_size, appliedHeat, first, last, dx, dt);
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


void DiffuseHeat(float* currentPoints, int size, double appliedHeat,float first, float last, double dx, double dt)
{
	//create temporary storage array
	int rank;
	int index;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	for(index = 1; index < size-1; index++)
	{
		printf("Process #%d receives %f, %f, %f, %f \n",rank, currentPoints[0], currentPoints[1], currentPoints[2], dt);
		currentPoints[index] = currentPoints[index] + (dt/dx*dx)*(currentPoints[index+1] - (2*currentPoints[index]) + currentPoints[index-1]);
	}
	printf("\n\n\nApplied Heat is %f\n\n\n", appliedHeat);
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
