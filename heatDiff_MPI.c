//Header files inclided and definitions.
//
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <stdio.h>
#define NUMPOINTS 1000
#define ENDTIME 30
#define DT 0.1
#define ROOM_TEMP 0

//Function declarations.
//
float* begin_computation(float* currentPoints, int w_rank, int w_size, double dx, double dt, int chunk_size);
void InitRod(float* array, int size, float roomTemp, double appliedHeat);
void DiffuseHeat(float* new_Buffer,float* currentPoints, int size, double dx, double dt, float first, float last, int rank, int w_size);
void PrintPoints(float* array, int size, double currentTime);

int main(/*int argc, char *argv[]*/)
{

	int w_rank;
	int w_size;
	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &w_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &w_size);
	double currentTime = 0;
	double mpi_time;
	float appliedHeat;
	double dx;
	double dt;
	int chunk_size;
	float* currentPoints = NULL;

	MPI_Barrier(MPI_COMM_WORLD);

	if(w_rank == 0)
	{
		printf("\n--------------------------------------------------------\n");
		printf("\n\t--> 1D - Heat Diffusion with MPI <--\n\n");
		printf("\tAuthors: Wade Pimenta & David Kroukamp\n");
		printf("\n--------------------------------------------------------\n");
		printf("\nInitialised with %d processors.\n", w_size);
		printf("\nBeginning computation ...\n");
		mpi_time = MPI_Wtime();
		currentPoints = (float*)malloc(NUMPOINTS*sizeof(float));
		srand(time(NULL));
		appliedHeat = rand()&100+1;
		InitRod(currentPoints, NUMPOINTS, ROOM_TEMP, appliedHeat);
		dx = currentPoints[1] - currentPoints[0];
		dt = DT;
		chunk_size = NUMPOINTS/w_size;

	}

	while(currentTime < ENDTIME)
	{
		if(w_rank == 0)
		{
//			PrintPoints(currentPoints, NUMPOINTS, currentTime);
		}
		begin_computation(currentPoints,w_rank, w_size, dx, dt, chunk_size);
		currentTime += DT;
	}
	MPI_Barrier(MPI_COMM_WORLD);

	if (w_rank == 0)
	{
		mpi_time = MPI_Wtime() - mpi_time;
	}

	MPI_Finalize();
	if(w_rank == 0)
	{
		PrintPoints(currentPoints, NUMPOINTS, ENDTIME);
		printf("\nCompleted in %f seconds wall clock time.\n\nData stored to 'HeatDiffusion.txt'.\n", mpi_time);
	}
	free(currentPoints);
	return 1;
}

float* begin_computation(float* currentPoints, int w_rank, int w_size, double dx, double dt, int chunk_size)
{
	float* result = (float*)malloc(NUMPOINTS*sizeof(float));
	FILE *output_file;
	double currentTime = 0.0;
	int n = NUMPOINTS;
	double timing;
	MPI_Status status;

	MPI_Bcast(&n, 1, MPI_INT, 0 , MPI_COMM_WORLD);
	MPI_Bcast(&dx, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&dt, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&chunk_size,1,MPI_INT,0,MPI_COMM_WORLD);

	float *process_recv_buffer = (float*)malloc((chunk_size)*sizeof(float));
	float *new_Buffer = (float*)malloc((chunk_size)*sizeof(float));
	float first; float last;
	int tag;

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
	DiffuseHeat(new_Buffer, process_recv_buffer, chunk_size, dx, dt, first, last, w_rank, w_size);

	MPI_Gather(new_Buffer, chunk_size, MPI_FLOAT, result, chunk_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

	if(w_rank == 0)
	{
		int index;
		for(index = 0; index < NUMPOINTS; index ++)
		{
			currentPoints[index] = result[index];
		}
	}
	free(process_recv_buffer); free(new_Buffer); free(result);
	return currentPoints;

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


void DiffuseHeat(float* new_Buffer, float* currentPoints, int size, double dx, double dt, float first, float last, int rank, int w_size)
{
	int index;
	int F = dt/(dx*dx);
	if(rank==0)
	{
		new_Buffer[0] = first;
		for(index = 1;index < size; index ++)
		{
			if(index == size-1)
			{
				new_Buffer[index]  = currentPoints[index] + 0.25*(last - (2*currentPoints[index]) + currentPoints[index-1]);
			}
			else
			{
				new_Buffer[index]  = currentPoints[index] + 0.25*(currentPoints[index+1] - (2*currentPoints[index]) + currentPoints[index-1]);
			}
		}
	}
	else if(rank == w_size-1)
	{
		new_Buffer[size-1] = last;
                for(index = 0;index < size-1; index ++)
                {
                        if(index == 0)
                        {
                        	new_Buffer[index] = currentPoints[index] + 0.25*(currentPoints[index+1] - (2*currentPoints[index]) + first);
                        }
                        else
                        {
                        	new_Buffer[index] = currentPoints[index] + 0.25*(currentPoints[index+1] - (2*currentPoints[index]) + currentPoints[index-1]);
                        }
                }

	}
	else
	{
		for(index = 0; index < size; index++)
		{
			if(index == 0)
                        {
                        	new_Buffer[index] = currentPoints[index] + 0.25*(currentPoints[index+1] - (2*currentPoints[index]) + first);
                        }
                        else if (index == size - 1)
                        {
                                new_Buffer[index] = currentPoints[index] + 0.25*(last - (2*currentPoints[index]) + currentPoints[index-1]);
                        }
			else
			{
                        	new_Buffer[index] = currentPoints[index] + 0.25*(currentPoints[index+1] - (2*currentPoints[index]) + currentPoints[index-1]);
			}
		}

	}
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
