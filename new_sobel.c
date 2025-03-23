#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <omp.h>

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
    for(int i = 0; i < 5000; i++)
    {
        for(int j = 0; j < 5000; j++)
        {
            fscanf(file, "%d", &num);
            data[i*5000+j] = num;
        }
    }
    fclose(file);
}

int main(int argc, char** argv)
{

    int col = 5000, row = 5000;
    int i, j;
    char dir[200];
    int rank, size;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int8_t* image_data = (int8_t*)calloc(col*row, sizeof(int8_t));
    //reading from file
    if(rank==0)
    {
        strcpy(dir, argv[1]);
        read_file(dir, image_data);
    }
    int8_t  gx, gy;
    
    for(int i = 1; i < col-2; i++)
    {
        for(int j = 1; j < row-2; j++)
        {
            
            gx = (image_data[(i-1)*col+(j-1)]*-1 	+ image_data[(i+1)*col+(j-1)]*1 
				+ image_data[(i-1)*col+(j)]*-2 	+ image_data[(i+1)*col+(j)]*2 
				+ image_data[(i-1)*col+(j+1)]*-1 	+ image_data[(i+1)*col+(j+1)]*1);

			gy = (image_data[(i-1)*col+(j-1)]*1 	+ image_data[i*col+(j-1)]*2 
				+ image_data[(i+1)*col+(j-1)]*1 	+ image_data[(i-1)*col+(j+1)]*-1 
				+ image_data[i*col+(j+1)]*-2 		+ image_data[(i+1)*col+(j+1)]*-1);
            
            
            image_data[i*col+j] = sqrt((gx*gx) + (gy*gy));
                //debugging
                //printf("%d", image_data[i*col+j]);
        }
    }
    //min max
    int min = 1000000, max = 0;

    for(i = 0; i < col; i++) {
		for(j = 0; j < row; j++) {
			if (image_data[i*col+j] < min) {
				min = image_data[i*col+j] ;
			}
			else if (image_data[i*col+j] > max) {
				max = image_data[i*col+j];
			}
		}
	}

    for(i = 0; i < col; i++) {
		for(j = 0; j < row; j++) {
			if (image_data[i*col+j]  > min + 70) {
				if (image_data[i*col+j] + 30 < 255)
                image_data[i*col+j] = image_data[i*col+j] + 30;
				else
                image_data[i*col+j] = 255;
			}
		}
	}


    //to lazy to make a function for the writing
    FILE* out;
    char* token = strtok(dir, ".");
	if (token != NULL) {
		strcat(token, "_filtered.txt");
		out = fopen(token, "wb");
	}
    out = fopen(dir, "wb");
    for(int i = 0;i < col; i++)
    {
        for(int j = 0; j < row; j++) 
        {
            fprintf(out, "%d     ", image_data[i*col+j]);
        }
        fprintf(out, "\n");
    }
    fclose(out);

    
    MPI_Finalize();
}   