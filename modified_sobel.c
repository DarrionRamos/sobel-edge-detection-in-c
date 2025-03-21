#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

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

void sobel_edge_detector(pgm* image, pgm* out_image) {
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
			/*if (matrix[i][j] > min + 70) {
				if (matrix[i][j] + 30 < 255)
					matrix[i][j] = matrix[i][j] + 30;
				else
					matrix[i][j] = 255;
			}*/
			double ratio = (double) (matrix[i][j] - min) / (max - min);
			//printf("Ratio: %d, Matrix value before: %d\n", ratio, matrix[i][j]);
			matrix[i][j] = ratio * 255;
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

int main(int argc, char **argv)
{
	pgm image, out_image;
	char dir[200];
	printf("Enter the file name: ");
	scanf("%s", dir);

	read_pgm_file(dir, &image);
	padding(&image);
	init_out_image(&out_image, image);
	sobel_edge_detector(&image, &out_image);

	min_max_normalization(&out_image, out_image.imageData);
	min_max_normalization(&out_image, out_image.gx);
	min_max_normalization(&out_image, out_image.gy);

	write_pgm_file(&out_image, dir, out_image.imageData, "_filtered.txt");
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
}
