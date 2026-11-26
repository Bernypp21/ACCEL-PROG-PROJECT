#ifndef STENCILS_CUH
#define STENCILS_CUH

#include <cuda_runtime.h>

#define IN_TILE_DIM 16
#define OUT_TILE_DIM (IN_TILE_DIM - 2)

// 1 = basic, 2 = shared input tile, 3 = shared output tile with halo.
__global__ void parallel_stencil_kernel(float *in, float *out, int N);
__global__ void shared_stencil_kernel(float *in, float *out, int N);
__global__ void shared_halo_stencil_kernel(float *in, float *out, int N);

#endif
