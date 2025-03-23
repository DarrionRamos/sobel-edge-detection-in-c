#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "mpi.h"
#include <omp.h>
#include <time.h> 
#include <unistd.h> 

int isspace(int argument);

typedef struct {
	int width;
	int height;
	int8_t **imageData;
	int8_t **gx;
	int8_t **gy;
} pgm;

void init_out_image( pgm* out, pgm image){
	int i, j;
	out->width = image.width;
	out->height = image.height;

	out->imageData = (int8_t**) calloc(out->height, sizeof(int8_t*));
	for(i = 0; i < out->height; i++) {
		out->imageData[i] = calloc(out->width, sizeof(int8_t));
	}

	out->gx = (int8_t**) calloc(out->height, sizeof(int*));
	for(i = 0; i < out->height; i++) {
		out->gx[i] = calloc(out->width, sizeof(int));
	}

	out->gy = (int8_t**) calloc(out->height, sizeof(int8_t*));
	for(i = 0; i < out->height; i++) {
		out->gy[i] = calloc(out->width, sizeof(int8_t));
	}

	for(i = 0; i < out->height; i++) {
		for(j = 0; j < out->width; j++) {
			out->imageData[i][j] = image.imageData[i][j];
			out->gx[i][j] = image.imageData[i][j];
			out->gy[i][j] = image.imageData[i][j];
		};
	}
}

void read_pgm_file(char* dir, pgm* image) {
	FILE* input_image;
	int i, j, num;

	input_image = fopen(dir, "rb");
	if (input_image == NULL) {
		printf("File could not opened!");
		return;
	}
	image->width = 5000;
	image->height = 5000;

	image->imageData = (int8_t**) calloc(image->height, sizeof(int8_t*));
	for(i = 0; i < image->height; i++) {
		image->imageData[i] = (int8_t*)calloc(image->width, sizeof(int8_t));
		if (image->imageData[i] == NULL) {
			printf("ERROR IN CALLOC\n");
			return;
		}
	}

	for (i = 0; i < image->height; i++) {
		for (j = 0; j < image->width; j++) {
			fscanf(input_image, "%d", &num);
//			printf("%d	", num);
			image->imageData[i][j] = num;
		}
//		printf("\n");
	}
	fclose(input_image);
}

void padding(pgm* image) {
	int i;
	for (i = 0; i < image->width; i++) {
		image->imageData[0][i] = 0;
		image->imageData[image->height - 1][i] = 0;
	}

	for (i = 0; i < image->height; i++) {
		image->imageData[i][0] = 0;
		image->imageData[i][image->width - 1] = 0;
	}
}



int convolution(pgm* image, int kernel[3][3], int row, int col) {
	int i, j, sum = 0;
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			sum += image->imageData[i + row][j + col] * kernel[i][j];
		}
	}
	return sum;
}

void sobel_edge_detector(pgm* image, pgm* out_image, int rows) {
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
	
	for (i = 1; i < image->height - 2; i++) {
		for (j = 1; j < image->width - 2; j++) {
			gx = convolution(image, mx, i, j);
			gy = convolution(image, my, i, j);
			out_image->imageData[i][j] = sqrt(gx*gx + gy*gy);
			out_image->gx[i][j] = gx;
			out_image->gy[i][j] = gy;
		}
	}
	
}

/*void sobel_edge_detector(pgm* image, pgm* out_image) {
	int i, j, gx, gy;
	int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
	int8_t *counts = (int8_t*)calloc(size, sizeof(int8_t));
	int8_t *displ = (int8_t*)calloc(size, sizeof(int8_t));
	int sum = 0;
	int8_t rows = (image->height-2)/size;
	int8_t rem = (image->height-2)%size;
	int height = image->height, width = image->width;
	int rows_proc = height/size;
	int cols = width;
	int comp_arr[height][width];
	int main[height][width];


	MPI_Scatter(main, (rows_proc*height), MPI_INT, comp_arr, rows_proc*height, MPI_INT, 0, MPI_COMM_WORLD);
	for(int i = 1; i < height+1; i++)
	{
		for (int j = 0; j < rows_proc + 1; j++) 
		{
			int gx = (comp_arr[i-1][j-1]*-1 + comp_arr[i+1][j-1]*1 + comp_arr[i-1][j]*-2 + comp_arr[i+1][j]*2 + comp_arr[i-1][j+1]*-1 + comp_arr[i+1][j+1]*1);
			int gy = (comp_arr[i-1][j-1]*1 + comp_arr[i][j-1]*2 + comp_arr[i+1][j-1]*1 
					+ comp_arr[i-1][j+1]*-1 + comp_arr[i][j+1]*-2 + comp_arr[i+1][j+1]*-1 + comp_arr[i+1][j+1]*1);
			//printf("%d", main[i][j]);
			int grad = gx*gx +
			image->imageData[i][j] = 
		}
	}

	//MPI_Scatter();
*/
	/*for(int i = 0; i < size; i++) {
		counts[i] = (rows + (i < rem? 1 : 0));
		displ[i] = sum;
		sum += counts[i];
	}*/

	/*int mx[3][3] = {
		{-1, 0, 1},
		{-2, 0, 2},
		{-1, 0, 1}
	};
	int my[3][3] = {
		{-1, -2, -1},
		{0, 0, 0},
		{1, 2, 1}
	};
	*/
	//#pragma omp parallel for private(i,j)
	

	//for (i = 1; i < image->height - 2; i++) {
	//	for (j = 1; j < image->width - 2; j++) {
			//gx = convolution(image, mx, i, j);
			//gy = convolution(image, my, i, j);
			//a[i*image->width + j] = sqrt(gx*gx + gy*gy);
	//	}
	//}
	
//}

void min_max_normalization(pgm* image, int8_t** matrix) {
	int min = 1000000, max = 0, i, j;

	for(i = 0; i < image->height; i++) {
		for(j = 0; j < image->width; j++) {
			if (matrix[i][j] < min) {
				min = matrix[i][j];
			}
			else if (matrix[i][j] > max) {
				max = matrix[i][j];
			}
		}
	}
	printf("min: %d, max: %d\n", min, max);
	for(i = 0; i < image->height; i++) {
		for(j = 0; j < image->width; j++) {
			// *** This code artificially increases the brightness of found edges since I was having trouble sometimes with the given normalization (what I used in the file I gave you) *** //
			// There could also be some issue with using int8_t instead of uint8_t but I have not tried the difference yet and I think int8_t looks good enough
			if (matrix[i][j] > min + 70) {
				if (matrix[i][j] + 30 < 255)
					matrix[i][j] = matrix[i][j] + 30;
				else
					matrix[i][j] = 255;
			}
			// *** This is the normalization that was done in the original repo *** //
			//double ratio = (double) (matrix[i][j] - min) / (max - min);
			//printf("Ratio: %d, Matrix value before: %d\n", ratio, matrix[i][j]);
			//matrix[i][j] = ratio * 255;
			//printf("Matrix value after: %d\n", matrix[i][j]);
		}
	}
}


void write_pgm_file(pgm* image, char dir[], int8_t** matrix, char name[]) {
	FILE* out_image;
	int i, j, count = 0;

	char* token = strtok(dir, ".");
	if (token != NULL) {
		strcat(token, name);
		out_image = fopen(token, "wb");
	}

	out_image = fopen(dir, "wb");
//	fprintf(out_image, "%d %d\n", image->width, image->height);

		for(i = 0; i < image->height; i++) {
			for(j = 0; j < image->width; j++) {
				fprintf(out_image,"%d     ", matrix[i][j]);
				/*if (count % 17 == 0)
					fprintf(out_image,"\n");
				else
					fprintf(out_image," ");
				count ++;*/
			}
			fprintf(out_image,"\n");
		}
	fclose(out_image);
}

int8_t* array2d_to_1d(pgm* twod)
{
	int8_t* one_d = (int8_t*)malloc(twod->height*twod->width*sizeof(int8_t));

	for(int i = 0; i < twod->height; i++)
	{
		for(int j = 0; j < twod->width; j++)
		{
			one_d[i*twod->width+j] = twod->imageData[i][j];
		}
	}
	return one_d;
}

int** array1d_to_2d(int* one_d, int height, int width)
{
	int** arr2d = (int**)malloc(width*sizeof(int*));

	for(int i = 0; i < height; i++)
	{
		arr2d[i] = (int*)malloc(height*sizeof(int));
		for(int j = 0; j < width; j++)
		{
			arr2d[i][j] = one_d[i*width+j];
		}
	}
	return arr2d;
}

int main(int argc, char **argv)
{
	MPI_Init(&argc,&argv);
	int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
	pgm image, out_image;
	char dir[200];
	printf("Enter the file name: ");
	strcpy(dir, argv[1]);

	read_pgm_file(dir, &image);
	padding(&image);
	init_out_image(&out_image, image);
	int proc_rows = 5000/size;

	//int8_t* local_image = (int8_t*)calloc(proc_rows*5000,sizeof(int8_t));
	//int8_t* image_1d = array2d_to_1d(image.imageData, 5000, 5000);

	int* global = (int*)calloc(5000*5000, sizeof(int));
	int8_t* image_1d = array2d_to_1d(&image);
	int* local_image = (int*)calloc(5000*proc_rows, sizeof(int));
	MPI_Scatter(image_1d, proc_rows*5000, MPI_INT,
				local_image, proc_rows*5000, MPI_INT, 0, MPI_COMM_WORLD);
	int* buff = (int*)calloc(5000*proc_rows, sizeof(int));
	int gx,  gy;
	for (int i = 1; i < proc_rows*5000+1; i++) {
		for (int j = 1; j < 5000 - 2 + 1; j++) {
			gx = (local_image[(i-1)*5000+(j-1)]*-1 	+ local_image[(i+1)*5000+(j-1)]*1 
				+ local_image[(i-1)*5000+(j)]*-2 	+ local_image[(i+1)*5000+(j)]*2 
				+ local_image[(i-1)*5000+(j+1)]*-1 	+ local_image[(i+1)*5000+(j+1)]*1);
			gy = (local_image[(i-1)*5000+(j-1)]*1 	+ local_image[i*5000+(j-1)]*2 
				+ local_image[(i+1)*5000+(j-1)]*1 	+ local_image[(i-1)*5000+(j+1)]*-1 
				+ local_image[i*5000+(j+1)]*-2 		+ local_image[(i+1)*5000+(j+1)]*-1);

				//printf("%d", sqrt(gx*gx+gy*gy));
			
		}
	}
	MPI_Gather(buff, proc_rows*5000, MPI_INT,
			global, proc_rows*5000, MPI_INT, 0, MPI_COMM_WORLD);

	if(rank == 0)
	{
		for(int i=0;i<5000;i++)
		{
			for(int j=0;j<5000;j++)
			{
				out_image.imageData[i][j] = buff[i*5000+j];
			}
		}
		min_max_normalization(&out_image, out_image.imageData);
		write_pgm_file(&out_image, dir, out_image.imageData, "_filtered.txt");
	}

	//sobel_edge_detector(&local_image, &out_image);
	//MPI_Gather(local.imageData, proc_rows, MPI_INT,
	//			image.imageData, proc_rows, MPI_INT, 0 , MPI_COMM_WORLD);

	//min_max_normalization(&out_image, out_image.imageData);
	//min_max_normalization(&out_image, out_image.gx);
	//min_max_normalization(&out_image, out_image.gy);


	//write_pgm_file(&out_image, dir, out_image.imageData, "_filtered.txt");

	printf("\nGradient saved: %s \n", dir);
//	write_pgm_file(&out_image, dir, out_image.gx, ".GX.pgm");
//	printf("Gradient X saved: %s \n", dir);
//	write_pgm_file(&out_image, dir, out_image.gy, ".GY.pgm");
//	printf("Gradient Y saved: %s \n", dir);

	free(image.imageData);
	free(out_image.imageData);
	free(out_image.gx);
	free(out_image.gy);
	return 0;
	MPI_Finalize();
}
