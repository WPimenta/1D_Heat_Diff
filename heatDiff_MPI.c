//Header files inclided and definitions.
//
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <stdio.h>
#define ROOM_TEMP 0
int NUMPOINTS = 1000000;
double ENDTIME = 1;
double DT = 0.1;
double ENDVALUES = 500;
//Function declarations.
//
float* begin_computation(float* currentPoints, int w_rank, int w_size, double dx, double dt, int chunk_size);
void InitRod(float* array, int size, float roomTemp, double appliedHeat);
void DiffuseHeat(float* new_Buffer,float* currentPoints, int size, double dx, double dt, float first, float last, int rank, int w_size);
void PrintPoints(float* array, int size, double currentTime);
int verify(float* currentPoints_serial, float* currentPoints_parallel, int size);
void serialDiffusion(float* currentPoints, float* result, double dx, double dt, double endTime);

double inputs[6][4] = {{12, 5, 0.1, 500}, {120, 5, 0.1, 500}, {1200, 5, 0.1, 500}, {12000, 5, 0.1, 500}, {120000, 5, 0.1, 500}, {1200000, 5, 0.1, 500}}; //Added this

int main()
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
	int testCase = 1;
	double sTime;
	float* currentPoints_serial = NULL;
	float* currentPoints_parallel = NULL;
	float* result = NULL;

	if(w_rank == 0)
	{
		printf("\n--------------------------------------------------------\n");
		printf("\n\t--> 1D - Heat Diffusion with MPI <--\n\n");
		printf("\tAuthors: Wade Pimenta & David Kroukamp\n");
		printf("\n--------------------------------------------------------\n");
		printf("\nInitialised with %d processors.\n", w_size);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	while (testCase <= 6)
	{
		if(w_rank == 0)
		{
			NUMPOINTS = (int)inputs[testCase-1][0]; //Added this
			ENDTIME = inputs[testCase-1][1]; //Added this
			DT = inputs[testCase-1][2]; //Added this
			ENDVALUES = inputs[testCase-1][3]; //Added this
			clock_t start, end;
			currentPoints_serial = (float*)malloc(NUMPOINTS*sizeof(float));
			currentPoints_parallel = (float*)malloc(NUMPOINTS*sizeof(float));
			result	 = (float*)malloc(NUMPOINTS*sizeof(float));
			appliedHeat = ENDVALUES;
			InitRod(currentPoints_parallel, NUMPOINTS, ROOM_TEMP, appliedHeat);
			InitRod(currentPoints_serial, NUMPOINTS, ROOM_TEMP, appliedHeat);
			dx = currentPoints_serial[1] - currentPoints_serial[0];
			dt = DT;
			chunk_size = NUMPOINTS/w_size;
			printf("\nHeating sample:\t%d.\n",testCase);
			start = clock();
			serialDiffusion(currentPoints_serial, result, dx, dt, ENDTIME);
			end = clock() - start;
			sTime = end/CLOCKS_PER_SEC;
		}

		MPI_Barrier(MPI_COMM_WORLD);
		if(w_rank == 0) mpi_time = MPI_Wtime();


		while(currentTime < ENDTIME)
		{
			begin_computation(currentPoints_parallel,w_rank, w_size, dx, dt, chunk_size);
			currentTime += DT;
		}
		MPI_Barrier(MPI_COMM_WORLD);

		if (w_rank == 0)
		{
			mpi_time = MPI_Wtime() - mpi_time;
			int verification = 0;
			printf("Verifying output.\n");
			verification = verify(currentPoints_serial, currentPoints_parallel, NUMPOINTS);
			if(verification == 0)
			{
				printf("Done.\n\n");
				printf("Serial\t:\t%f seconds.\n", sTime);
				printf("MPI\t:\t%f seconds.\n", mpi_time);
				printf("Speedup\t:\t%f.\n", sTime/mpi_time);
			}
			else if (verification == 1)
			{
				printf("Error: Serial and Parallel computation mismatch.");
			}
		}

		if(w_rank == 0)
		{
			//PrintPoints(currentPoints_parallel, NUMPOINTS, ENDTIME);
			testCase++; //Added this
			MPI_Bcast(&testCase, 1, MPI_INT, 0 , MPI_COMM_WORLD); //Added this
		}
		free(currentPoints_serial); free(currentPoints_parallel); free(result);
	}
	MPI_Finalize();	
	return 1;
}

int verify(float* currentPoints_serial, float* currentPoints_parallel, int size)
{
	int check = 0;
	int index;
	for(index = 0; index < size; index ++)
	{
		if(currentPoints_serial[index] != currentPoints_parallel[index]) check = 1;
	}
	return check;
}

void serialDiffusion(float *currentPoints, float* result, double dx, double dt, double endTime)
{
	double currentTime = 0;
	int index;
	while(currentTime < endTime)
	{
		for(index = 1; index < NUMPOINTS -1; index ++)
		{
			result[index] = currentPoints[index] + 0.25*(currentPoints[index+1]-2*(currentPoints[index])+currentPoints[index-1]);
		}
		for(index = 1; index < NUMPOINTS -1; index++)
		{
			currentPoints[index] = result[index];
		}
		currentTime += dt;
	}	
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
