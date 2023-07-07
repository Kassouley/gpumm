#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "tab.h"


void init_tab2d_random(unsigned int n, double** tab)
{
    for (unsigned int i = 0; i < n; i++)
    {
        for (unsigned int j = 0; j < n; j++)
        {
            (*tab)[i*n+j] = (double)rand() / (double)RAND_MAX;
        }
    }
}

