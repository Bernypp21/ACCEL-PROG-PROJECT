#include <cuda_runtime.h>
#include "stencils.cuh"

// Next step: one 16x16 output tile with a one-pixel halo.
__global__ void shared_halo_stencil_kernel(float *in, float *out, int N)
{
    // Extra rows and columns hold neighbors just outside the block.
    __shared__ float tile[IN_TILE_DIM + 2][IN_TILE_DIM + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * IN_TILE_DIM + tx;
    int row = blockIdx.y * IN_TILE_DIM + ty;

    // Shared-memory coordinates are shifted by one for the halo.
    int sx = tx + 1;
    int sy = ty + 1;

    // Step 1: Every thread loads its center pixel.
    if (row < N && col < N)
        tile[sy][sx] = in[row * N + col];
    else
        tile[sy][sx] = 0.0f;

    // Step 2: Edge threads also load one neighboring halo pixel.
    if (tx == 0)
        tile[sy][0] = (row < N && col > 0)
            ? in[row * N + col - 1] : 0.0f;

    if (tx == IN_TILE_DIM - 1)
        tile[sy][IN_TILE_DIM + 1] = (row < N && col + 1 < N)
            ? in[row * N + col + 1] : 0.0f;

    if (ty == 0)
        tile[0][sx] = (row > 0 && col < N)
            ? in[(row - 1) * N + col] : 0.0f;

    if (ty == IN_TILE_DIM - 1)
        tile[IN_TILE_DIM + 1][sx] = (row + 1 < N && col < N)
            ? in[(row + 1) * N + col] : 0.0f;

    // Step 3: Do not read shared memory until all threads finish loading.
    __syncthreads();

    // Step 4: Average the center and its four neighbors.
    // The outside image border stays zero, as in the basic kernel.
    if (row > 0 && row < N - 1 && col > 0 && col < N - 1)
    {
        int idx = row * N + col;
        out[idx] = 0.2f * (
            tile[sy][sx] +
            tile[sy][sx - 1] +
            tile[sy][sx + 1] +
            tile[sy - 1][sx] +
            tile[sy + 1][sx]
        );
    }
}
