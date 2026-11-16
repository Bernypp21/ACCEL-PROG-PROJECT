

#ifndef STENCILS_CUH
#define STENCILS_CUH

#include <cuda_runtime.h>


// the basic parrallel stencil kernel call
__global__
void parallel_stencil_kernel(float *in, float *out, int N);




#endif