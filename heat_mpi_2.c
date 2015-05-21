//1-D Heat Diffusion Using MPI
//authors: Wade Pimenta & David Kroukamp
//
//Compile with:	mpicc heatDiff_MPI.c -o heatDiff_MPI
//Run with:	mpiexec -n [num_processors] ./heatDiff_MPI
//


//Header files and constants.
//
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <stdio.h>
#include <string.h>
#define MPI_OUTPUT "mpi_output.txt"
#define MPI_INPUT "mpi_input.txt"
#define ROOM_TEMP 0

//Global declarations.
//
float* begin_computation(float* result, float* currentPoints, int w_rank, int w_size, double dx, double dt, int chunk_size);
void InitRod(float* array, int size, float roomTemp, double appliedHeat);
void DiffuseHeat(float* new_Buffer,float* currentPoints, int size, double dx, double dt, float first, float last, int rank, int w_size);
void PrintPoints(float* array, int size, double currentTime);
void serialDiffusion(float* currentPoints,float* result,double dx,double dt,double endTime);
int verify (float* currentPoint_serial, float* currentPoints_parallel, int size);
void ProcessOutput(float* array, int testCase, float time);
int NUMPOINTS;
double ENDTIME;
double DT;
double ENDVALUES;

int main()
{
	int w_rank;
	int w_size;

	double sTime;
	double currentTime = 0;
	double mpi_time;
	float appliedHeat;
	double dx;
	double dt;
	int chunk_size;
	int testCase = 1;
	char str[70];
	FILE *p;	

	printf("\n--------------------------------------------------------\n");
	printf("\n\t--> 1D - Heat Diffusion with MPI <--\n\n");
	printf("\tAuthors: Wade Pimenta & David Kroukamp\n");
	printf("\n--------------------------------------------------------\n\n\n");
	remove(MPI_OUTPUT);
	if((p=fopen(MPI_INPUT,"r"))==NULL)
	{
		printf("Error: Unable to open mpi_input.txt");
		exit(1);
	}
	while(fgets(str,70,p)!=NULL)
	{
		char *pch;
		pch = strtok(str, " ");
		NUMPOINTS = atof(pch);
		pch = strtok(NULL, " ");
		ENDTIME = atof(pch);
		pch = strtok(NULL, " ");
		DT = atof(pch);
		pch = strtok(NULL, " ");
		ENDVALUES = atof(pch);
		printf("%d, %f, %f, %f <-----", NUMPOINTS, ENDTIME, DT, ENDVALUES);
		
		float* currentPoints_serial = 0;	
		currentPoints_parallel = (float*)malloc(NUMPOINTS*sizeof(float));
		float* currentPoints_parallel = 0;	
		currentPoints_serial = (float*)malloc(NUMPOINTS*sizeof(float));
		float* result = 0;
		result = (float*)malloc(NUMPOINTS*sizeof(float));
		sTime = 0;
		clock_t start, end;
		appliedHeat = ENDVALUES;
		InitRod(currentPoints_parallel, NUMPOINTS, ROOM_TEMP, appliedHeat);
		InitRod(currentPoints_serial, NUMPOINTS, ROOM_TEMP, appliedHeat);
		dx = currentPoints_serial[1] - currentPoints_serial[0];
		dt = DT;
		
		printf("Heating sample:\t%d.\n", testCase);
		start = clock();
		serialDiffusion(currentPoints_serial, result, dx, dt, ENDTIME);
		end = clock() - start;
		sTime = end/CLOCKS_PER_SEC;

		MPI_Init(NULL, NULL);
		MPI_Comm_rank(MPI_COMM_WORLD, &w_rank);
		MPI_Comm_size(MPI_COMM_WORLD, &w_size);
		chunk_size = NUMPOINTS/w_size;
		
		MPI_Bcast(&NUMPOINTS, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&ENDTIME, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(&DT, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(&ENDVALUES, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		
		MPI_Bcast(&currentPoints_parallel, NUMPOINTS, MPI_FLOAT,0, MPI_COMM_WORLD);

		/*while(currentTime < ENDTIME)
		{
			currentTime += DT;
			begin_computation(result, currentPoints_parallel, w_rank, w_size, dx, dt, chunk_size);
			//xPrintPoints(currentPoints_parallel, NUMPOINTS, currentTime);
		}*/
		free(currentPoints_serial);
		free(result);
		free(currentPoints_parallel);
		MPI_Finalize();
	}
	printf("\n -------->\tData Stored to mpi_output.txt\t<--------\n");
	fclose(p);
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

void serialDiffusion(float* currentPoints, float* result, double dx, double dt, double endTime)
{
	double currentTime = 0.0;
	int index;
	while(currentTime < endTime)
	{
		for(index = 1; index < NUMPOINTS - 1; index ++)
		{
			result[index] = currentPoints[index] + 0.25*(currentPoints[index+1] - 2*(currentPoints[index]) + currentPoints[index-1]);
		}
		for(index = 1; index < NUMPOINTS -1; index++)
		{
			currentPoints[index] = result[index];
		}
		currentTime += dt;
	}
}

float* begin_computation(float* result,float* currentPoints, int w_rank, int w_size, double dx, double dt, int chunk_size)
{
	double currentTime = 0.0;
	int n = NUMPOINTS;
	double timing;
	MPI_Status status;

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
	free(process_recv_buffer); free(new_Buffer);
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
	array[size-1] = appliedHeat;
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

void ProcessOutput(float* array, int testCase, float time)
{
	int index;
	FILE *f = fopen(MPI_OUTPUT, "a");
	if(f == NULL)
	{
		printf("Error: Could not open mpi_output.txt");
		exit(1);
	}
	fprintf(f, "Runtime for test case %d with %d points:\n", testCase, NUMPOINTS);
	fprintf(f, "%f\n", time);
	if(NUMPOINTS <= 15)
	{
		fprintf(f, "Resultant temperatures:\n");
		for(index = 0; index < NUMPOINTS; index++)
		{
			fprintf(f, "%0.2f ", array[index]);
		}
	}
	fprintf(f,"\n\n");
	fclose(f);
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

