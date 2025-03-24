//#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
//#include <omp.h>

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
    /*for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < col; j++)
        {
            fscanf(file, "%d", &num);
            data[i*col+j] = num;
            printf("%d     ", num);
	}
	printf("\n");
    }*/
    for (int i = 0; i < row*col; i++) {
	fscanf(file, "%d", &num);
	data[i] = num;
	printf("%d,%d     ", data[i], num);
	if (i % row == 0)
		printf("\n");
    }
    fclose(file);
}

void padding(int* image_data) {
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

int convolution(int* image_data, int kernel[3][3], int row_l, int col_l) {
	int i, j, sum = 0;
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			int index = i*3 + row_l + j + col_l;
			sum += image_data[index] * kernel[i][j];
			if (i+row_l == 1 && j+col_l == 1)
				printf("Image data at: i: %d, j: %d, is: %d\n", i+row_l, j+col_l, image_data[index]);
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

	for (i = 1; i < row/size - 2; i++) {
		for (j = 1; j < col - 2; j++) {
			gx = convolution(in_image, mx, i*5000, j);
			gy = convolution(in_image, my, i*5000, j);
			out_image[i*(col)+j] = sqrt(gx*gx + gy*gy);
			//if (i == 4)
			//	printf("i: %d, j: %d, gx: %d, gy: %d\n", i, j, gx, gy);
		}
	}
}

void min_max_normalization(int* image_data) {
    int min = 0, max = 255;

    /*for(int i = 0; i < row/size; i++) {
		for(int j = 0; j < col; j++) {
			if (image_data[i*col+j] < min) {
				min = image_data[i*col+j] ;
			}
			else if (image_data[i*col+j] > max) {
				max = image_data[i*col+j];
			}
		}
	}*/

    for(int i = 0; i < row/size; i++) {
		for(int j = 0; j < col; j++) {
			if (image_data[i*col+j]  > min + 70) {
				if (image_data[i*col+j] + 30 < 255)
			                image_data[i*col+j] = image_data[i*col+j] + 30;
				else
			                image_data[i*col+j] = 255;
			}
		}
	}
}

void write_file(int* out_data, char dir[]) {
    FILE* out_file;
    char* token = strtok(dir, ".");
    if (token != NULL) {
	strcat(token, "_filtered.txt");
	out_file = fopen(token, "wb");
    }
    out_file = fopen(dir, "wb");
    for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < col; j++)
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
    //MPI_Init(&argc,&argv);
    //MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //MPI_Comm_size(MPI_COMM_WORLD, &size);
    rank = 0;
    size = 1;
    int* image_data = (int*)calloc(col*row, sizeof(int));
    int* out_data = (int*)calloc(col*row, sizeof(int));

    if(rank==0)
    {
        strcpy(dir, argv[1]);
        //read_file(dir, image_data);
	FILE* file;
    file = fopen("input.txt", "rb");
    if(file == NULL)
    {
        printf("File could not open\n");
        return -1;
    }
    printf("opened file: %s\n", dir);
    int num;
    /*for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < col; j++)
        {
            fscanf(file, "%d", &num);
            data[i*col+j] = num;
            printf("%d     ", num);
        }
        printf("\n");
    }*/
    for (int i = 0; i < row*col; i++) {
        fscanf(file, "%d", &num);
        image_data[i] = num;
        printf("%d,%d     ", image_data[i], num);
        if (i % row == 0)
                printf("\n");
    }
    fclose(file);

    }

//    padding(image_data);

    sobel_edge_detector(image_data, out_data);

    //min_max_normalization(out_data);

    if (rank == 0)
    	write_file(out_data, dir);

    free(image_data);
    free(out_data);

    //MPI_Finalize();
}
