#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kernel.h"
#include "tab.h"

#define OUTOUT_FILE "output_check.txt"

int main(int argc, char **argv)
{
    int n;
    char* file_name = NULL;
    FILE * output = NULL;

    if (argc != 2 && argc != 3) 
    {
        fprintf (stderr, "Usage: %s <problem size> [file name]\n", argv[0]);
        return 1;
    }
    else
    {
        n = atoll(argv[1]);
        file_name = (char*)malloc(256*sizeof(char));
        if (argc == 2)
            strcpy(file_name, OUTOUT_FILE);
        else if (argc == 3)
            strcpy(file_name, argv[2]);
    }
    
    int size = n * n * sizeof(double);

    double *a = (double*)malloc(size);
    double *b = (double*)malloc(size);
    double *c = (double*)malloc(size);

    srand(0);
    init_tab2d_random(n, &b);
    init_tab2d_random(n, &c);

    kernel(n, a, b, c);

    output = fopen(file_name, "w");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            fprintf(output, "%f ", a[i*n+j]);
        }
        fprintf(output, "\n");
    }
    fclose(output);
    
    free(file_name);
    free(a);
    free(b);
    free(c);
}
