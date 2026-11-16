

#include <cuda_runtime.h>

#include "stencils.cuh"



// a basic parallel stencil
__global__
void parallel_stencil_kernel(float *in, float *out, int N)
{
    // get the current thread location
    int inCol = blockIdx.x * blockDim.x + threadIdx.x;
    int inRow = blockIdx.y * blockDim.y + threadIdx.y;

    // basic parallel stencil // check for out of bounds
    if (inCol >= 1 && inCol < (N -1) && inRow >= 1 && inRow < (N - 1))
    {
        int idx = inRow * N + inCol;

        //update the out array by geting the center,left,right,upbove, and below make a "+"
        // and getting the average value
        out[idx] = 0.2f * ( 
            in[idx] +
            in[idx - 1] +
            in[idx + 1] +
            in[idx - N] +
            in[idx + N]
        );
    }
}

