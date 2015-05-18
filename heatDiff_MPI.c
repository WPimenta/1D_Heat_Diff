//Header files inclided and definitions.
//
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <stdio.h>
#define NUMPOINTS 12
#define ENDTIME 1
#define DT 0.1
#define ROOM_TEMP 0

//Function declarations.
//
void InitRod(float* array, int size, float roomTemp, double appliedHeat);
float* DiffuseHeat(float* currentPoints, int size, float apliedHeat, double dx, double dt, float first, float last, int rank);
void PrintPoints(float* array, int size, double currentTime);

int main()
{
	double currentTime = 0.0;
	int n;
	int chunk_size;
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


	float* currentPoints = NULL;

	if(w_rank == 0)
	{
		float first;
		float last;
		srand(time(NULL));
		currentPoints = (float*)malloc(NUMPOINTS*sizeof(float));
		appliedHeat = rand()&100+1;
		InitRod(currentPoints, NUMPOINTS, ROOM_TEMP, appliedHeat);
		dx = currentPoints[1] - currentPoints[0];
		n = NUMPOINTS;
		dt = DT;
		chunk_size = n/w_size;
		printf(" Chunk == %d\n\n", chunk_size);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	timing -= MPI_Wtime();

	MPI_Bcast(&appliedHeat, 1, MPI_FLOAT, 0 , MPI_COMM_WORLD);
	MPI_Bcast(&n, 1, MPI_INT, 0 , MPI_COMM_WORLD);
	MPI_Bcast(&dx, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&dt, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&chunk_size,1,MPI_INT,0,MPI_COMM_WORLD);

	float *process_recv_buffer = (float*)malloc((chunk_size)*sizeof(float));
	float first; float last;
	int tag;

	while(currentTime < ENDTIME)
	{
		MPI_Scatter(currentPoints, chunk_size, MPI_FLOAT, process_recv_buffer, chunk_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

		if(w_rank ==0)
		{
			first = process_recv_buffer[0];
		}
		if(w_rank == w_size-1)
		{
			last = process_recv_buffer[chunk_size-1];
		}
		if(w_rank > 0)
		{
			tag = 1;
			MPI_Send(&process_recv_buffer[0], 1, MPI_FLOAT, w_rank-1, tag, MPI_COMM_WORLD);
		}
		if(w_rank < w_size-1)
                {
			tag = 1;
                        MPI_Recv(&last, 1, MPI_FLOAT, w_rank+1, tag, MPI_COMM_WORLD, &status);
		}
		if(w_rank < w_size-1)
		{
			tag = 2;
			MPI_Send(&process_recv_buffer[chunk_size-1], 1, MPI_FLOAT, w_rank+1, tag, MPI_COMM_WORLD);
		}
		if(w_rank > 0)
		{
			tag = 2;
			MPI_Recv(&first, 1, MPI_FLOAT, w_rank-1, tag, MPI_COMM_WORLD, &status );
		}
		printf("%d\t%f\t%f\n",w_rank,first,last);
		break;

		DiffuseHeat(process_recv_buffer, chunk_size, appliedHeat, dx, dt, first, last, w_rank);
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


float* DiffuseHeat(float* currentPoints, int size, float appliedHeat, double dx, double dt, float first, float last, int rank)
{
	int index;
	int F = dt/(dx*dx);
	if(rank==0)
	{
		result[0] = first;
		for(index = 1;index < size; index ++)
		{
			if(index == size-1)
			{
				result[index] = currentPoints[index] + 0.25*(last - (2*currentPoints[index]) + currentPoints[index-1]);
			}
			else
			{
				result[index] = currentPoints[index] + 0.25*(currentPoints[index+1] - (2*currentPoints[index]) + currentPoints[index-1]);
			}
		}
	}
	else if(rank == size-1)
	{
		result[size-1] = last;
                for(index = 0;index < size-1; index ++)
                {
                        if(index == 0)
                        {
                                result[index] = currentPoints[index] + 0.25*(currentPoints[index+1] - (2*currentPoints[index]) + first);
                        }
                        else
                        {
                                result[index] = currentPoints[index] + 0.25*(currentPoints[index+1] - (2*currentPoints[index]) + currentPoints[index-1]);
                        }
                }

	}
	else
	{
		for(index = 0; index < size; index++)
		{
			if(index == 0)
                        {
                                result[index] = currentPoints[index] + 0.25*(currentPoints[index+1] - (2*currentPoints[index]) + first);
                        }
                        else if (index == size - 1)
                        {
                                result[index] = currentPoints[index] + 0.25*(last - (2*currentPoints[index]) + currentPoints[index-1]);
                        }
			else
			{
                                result[index] = currentPoints[index] + 0.25*(currentPoints[index+1] - (2*currentPoints[index]) + currentPoints[index-1]);
			}
		}

	}
	return result;
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
