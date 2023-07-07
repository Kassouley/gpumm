#include <stdio.h>
#include <openacc.h>
#include "kernel.h"
#include "device_acc.h"

#ifdef OPENACC
void kernel (unsigned int n, double* a, const double* b, const double* c)
{
    #pragma acc data copyin(b[0:n*n], c[0:n*n]) copyout(a[0:n*n])
    {
        #pragma acc loop independent gang vector
        for(unsigned int i = 0; i < n; i++)
        {
            for(unsigned int j = 0; j < n; j++)
            {
                a[i*n+j] = 0;
                for(unsigned int k = 0; k < n; k++)
                {
                    a[i*n+j] += b[i*n+k] * c[k*n+j];
                }
            }
        }
    }
}
#endif