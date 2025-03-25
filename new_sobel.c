#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <omp.h>

int isspace(int argument);

int rank, size;
int col = 5000, row = 5000;

void read_file(char* dir, int* data)
{
    FILE* file;
    file = fopen(dir, "rb");
    if(file == NULL)
    {
        printf("File could not open\n");
        return;
    }
    printf("opened file: %s\n", dir);
    int num;
    for (int i = 0; i < row*col; i++) {
	fscanf(file, "%d", &num);
	data[i] = num;
	printf("%d,%d     ", data[i], num);
	if (i % row == 0)
		printf("\n");
    }
    fclose(file);
}

int convolution(int* image_data, int kernel[3][3], int row_l, int col_l) {
	int i, j, sum = 0;
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			int index = i*3 + row_l + j + col_l;
			sum += image_data[index] * kernel[i][j];
		}
	}
	return sum;
}

void sobel_edge_detector(int* in_image, int* out_image) {
	int i, j, gx, gy;
	int mx[3][3] = {
		{-1, 0, 1},
		{-2, 0, 2},
		{-1, 0, 1}
	};
	int my[3][3] = {
		{-1, -2, -1},
		{0, 0, 0},
		{1, 2, 1}
	};
	#pragma omp parallel for private(i,j)
	for (i = 1; i < row/size - 2; i++) {
		for (j = 1; j < col - 2; j++) {
			gx = convolution(in_image, mx, i*5000, j);
			gy = convolution(in_image, my, i*5000, j);
			out_image[i*(col)+j] = sqrt((gx*gx + gy*gy));
		}
	}
}	

void write_file(int* out_data, char dir[]) {
    FILE* out_file;
	int i, j;
    char* token = strtok(dir, ".");
    if (token != NULL) {
	strcat(token, "_filtered.txt");
	out_file = fopen(token, "wb");
    }
    out_file = fopen(dir, "wb");
	//#pragma omp parallel for private(i,j)
    for(i = 0; i < row; i++)
    {
        for(j = 0; j < col; j++)
        {
            fprintf(out_file, "%d     ", (u_int8_t)out_data[i*col+j]);
        }
        fprintf(out_file, "\n");
    }
    fclose(out_file);
}

int main(int argc, char** argv)
{
    int i, j;
    char dir[200];
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int* image_data = (int*)calloc(col*row, sizeof(int));
    int* out_data = (int*)calloc(col*row, sizeof(int));

	int *count = (int*)calloc(row*col, sizeof(int));
	int *displ = (int*)calloc(row*col, sizeof(int));
	int num_per_proc = row/size;
	int remainder = row%size;
	int rows_per_proc = num_per_proc + (rank < remainder?1:0);
	//printf("%d %d %d", num_per_proc, remainder, rows_per_proc);

	int *local_buf = (int*)calloc(rows_per_proc*col, sizeof(int));
	int *out_buf = (int*)calloc(rows_per_proc*col, sizeof(int));

	clock_t start = clock();

    if(rank==0)
    {
		strcpy(dir, argv[1]);
		FILE* file;
		file = fopen("input.txt", "rb");
		if(file == NULL)
		{
			printf("File could not open\n");
			return -1;
		}
		printf("opened file: %s\n", dir);
		int num;
		//#pragma omp parallel for private(i)
		for (int i = 0; i < row*col; i++) {
			fscanf(file, "%d", &num);
			image_data[i] = num;
			//printf("%d ", image_data[i]);
		}
		fclose(file);
	}
	int temp = 0;

	MPI_Scatter(image_data, num_per_proc*col, MPI_INT, local_buf, num_per_proc*col, MPI_INT, 0, MPI_COMM_WORLD);
    sobel_edge_detector(local_buf, out_buf);
	MPI_Gather(out_buf, num_per_proc*col, MPI_INT, out_data, num_per_proc*col, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    	write_file(out_data, dir);

	clock_t end = clock();
	if(rank==0)
	printf("Program time: %.5f sec\n", ((double)(end-start))/CLOCKS_PER_SEC);
    free(image_data);
    free(out_data);

    MPI_Finalize();
}
