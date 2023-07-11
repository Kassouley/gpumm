#include <stdio.h>
#include <openacc.h>
#include "kernel.h"
#include "device_acc.h"

#ifdef OPENACC
void kernel (unsigned int n, double* a, const double* b, const double* c)
{
    // #pragma acc data copyin(b[0:n*n], c[0:n*n]) copyout(a[0:n*n])
    // {
        // #pragma acc kernels
        // #pragma acc loop independent gang vector
    #pragma acc data copyin(b[0:n*n], c[0:n*n]) copyout(a[0:n*n])
    {
        # pragma acc region
        {
            # pragma acc loop independent vector(32) 
            for(unsigned int i = 0; i < n; i++)
            {
                # pragma acc loop independent vector(32) 
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
}
#endif

#ifdef OPENACC_WO_DT
void kernel (unsigned int n, double* a, const double* b, const double* c)
{
    #pragma acc data deviceptr(a, b, c)
    {
        #pragma acc region
        {
            #pragma acc loop independent vector(32) 
            for(unsigned int i = 0; i < n; i++)
            {
                #pragma acc loop independent vector(32) 
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
}
#endif