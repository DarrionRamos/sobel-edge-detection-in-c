#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <omp.h>

int rank, size;
int i, j;
int col = 5000, row = 5000;

void read_file(char* dir, int8_t* data )
{
    FILE* file;
    file = fopen(dir, "rb");
    if(file == NULL)
    {
        printf("File could not open\n");
        return;
    }
    int num;
    for(int i = 0; i < col; i++)
    {
        for(int j = 0; j < row; j++)
        {
            fscanf(file, "%d", &num);
            data[i*row+j] = num;
            printf("%d ", num);
        }
    }
    fclose(file);
}

void padding(int8_t* image_data) {
	int i;
	for (i = 0; i < row; i++) {
		image_data[0*row+i] = 0;
		image_data[(col - 1)*row+i] = 0;
	}

	for (i = 0; i < col; i++) {
		image_data[0*col+i] = 0;
		image_data[(row - 1)*col+i] = 0;
	}
}

int convolution(int8_t* image_data, int kernel[3][3], int row_l, int col_l) {
	int i, j, sum = 0;
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			sum += image_data[(i+row_l)*3+(j+col_l)] * kernel[i][j];
			//printf("Image data at: i: %d, j: %d, is: %d\n", i+row_l, j+col_l, image_data[(i+row_l)*3+(j*col_l)]);
		}
	}
	return sum;
}

void sobel_edge_detector(int8_t* in_image, int8_t* out_image) {
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

	for (i = 1; i < col/size - 2; i++) {
		for (j = 1; j < row - 2; j++) {
			gx = convolution(in_image, mx, i, j);
			gy = convolution(in_image, my, i, j);
			out_image[i*(row-2)+j] = sqrt(gx*gx + gy*gy);
			//if (i == 4)
				//printf("i: %d, j: %d, gx: %d, gy: %d\n", i, j, gx, gy);
		}
	}
}

void min_max_normalization(int8_t* image_data) {
	int min = 1000000, max = 0;

    for(i = 0; i < col/size; i++) {
		for(j = 0; j < row; j++) {
			if (image_data[i*row+j] < min) {
				min = image_data[i*row+j] ;
			}
			else if (image_data[i*row+j] > max) {
				max = image_data[i*row+j];
			}
		}
	}

    for(i = 0; i < col/size; i++) {
		for(j = 0; j < row; j++) {
			if (image_data[i*row+j]  > min + 70) {
				if (image_data[i*row+j] + 30 < 255)
                image_data[i*row+j] = image_data[i*row+j] + 30;
				else
                image_data[i*row+j] = 255;
			}
		}
	}
}

void write_file(int8_t* out_data, char dir[]) {
    FILE* out_file;
    char* token = strtok(dir, ".");
	if (token != NULL) {
		strcat(token, "_filtered.txt");
		out_file = fopen(token, "wb");
	}
    out_file = fopen(dir, "wb");
    for(int i = 0; i < col; i++)
    {
        for(int j = 0; j < row; j++) 
        {
            fprintf(out_file, "%d     ", out_data[i*row+j]);
        }
        fprintf(out_file, "\n");
    }
    fclose(out_file);
}

int main(int argc, char** argv)
{
    char dir[200];
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int8_t* image_data = (int8_t*)calloc(col*row, sizeof(int8_t));
    int8_t* out_data = (int8_t*)calloc(col*row, sizeof(int8_t));

    //if(rank==0)
    //{
    //strcpy(dir, argv[1]);
    read_file("input.txt", image_data);
	//}

    sobel_edge_detector(image_data, out_data);

    min_max_normalization(image_data);

    //if (rank == 0)
    write_file(out_data, dir);

    free(image_data);
    free(out_data);

    MPI_Finalize();
}
