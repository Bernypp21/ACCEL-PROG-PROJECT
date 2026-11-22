#include <cuda_runtime.h>
#include "stencils.cuh"

// Five-point stencil: average a pixel with its four neighbors.
__global__ void shared_stencil_kernel(float *in, float *out, int N)
{
    // Fast memory shared by all threads in this block.
    __shared__ float tile[IN_TILE_DIM][IN_TILE_DIM];

    // Position of this thread inside its 16x16 block.
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Each block produces a 14x14 part of the image.
    // Subtract 1 so the outside threads load the halo (border).
    int col = blockIdx.x * OUT_TILE_DIM + tx - 1;
    int row = blockIdx.y * OUT_TILE_DIM + ty - 1;

    // Step 1: Each thread copies one pixel into shared memory.
    // Use zero when a pixel is outside the image.
    if (row >= 0 && row < N && col >= 0 && col < N)
        tile[ty][tx] = in[row * N + col];
    else
        tile[ty][tx] = 0.0f;

    // Step 2: Wait until everyone has loaded their pixel.
    __syncthreads();

    // Step 3: Only the middle 14x14 threads produce output.
    // Also skip the outermost image border, like the basic kernel.
    if (tx > 0 && tx < IN_TILE_DIM - 1 &&
        ty > 0 && ty < IN_TILE_DIM - 1 &&
        row > 0 && row < N - 1 &&
        col > 0 && col < N - 1)
    {
        int idx = row * N + col;

        // Read center, left, right, up, and down from shared memory.
        out[idx] = 0.2f * (
            tile[ty][tx] +
            tile[ty][tx - 1] +
            tile[ty][tx + 1] +
            tile[ty - 1][tx] +
            tile[ty + 1][tx]
        );
    }
}
