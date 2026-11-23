#ifndef STENCILS_CUH
#define STENCILS_CUH

#include <cuda_runtime.h>

// Each CUDA block has 16x16 threads.
#define IN_TILE_DIM 16

// Starter shared kernel: two halo rows/columns leave 14x14 outputs.
#define OUT_TILE_DIM (IN_TILE_DIM - 2)

// 1 = basic, 2 = textbook shared tile, 3 = shared tile with extra halo loads.
__global__ void parallel_stencil_kernel(float *in, float *out, int N);
__global__ void shared_stencil_kernel(float *in, float *out, int N);
__global__ void shared_halo_stencil_kernel(float *in, float *out, int N);

#endif
