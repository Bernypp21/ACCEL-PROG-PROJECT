#include "stencils.cuh"

// Type 2: textbook tile. 16x16 threads load pixels;
// the middle 14x14 threads produce output.
__global__ void shared_stencil_kernel(float *in, float *out, int N)
{
    __shared__ float tile[IN_TILE_DIM][IN_TILE_DIM];
    int tx = threadIdx.x, ty = threadIdx.y;
    int col = blockIdx.x * OUT_TILE_DIM + tx - 1;
    int row = blockIdx.y * OUT_TILE_DIM + ty - 1;

    // Load one pixel per thread, using zero outside the image.
    tile[ty][tx] = (row >= 0 && row < N && col >= 0 && col < N)
        ? in[row * N + col] : 0.0f;
    __syncthreads(); // Wait before other threads read the tile.

    // Halo threads only load; interior threads calculate.
    if (tx > 0 && tx < IN_TILE_DIM - 1 &&
        ty > 0 && ty < IN_TILE_DIM - 1 &&
        row > 0 && row < N - 1 && col > 0 && col < N - 1)
    {
        out[row * N + col] = 0.2f * (
            tile[ty][tx] + tile[ty][tx - 1] + tile[ty][tx + 1] +
            tile[ty - 1][tx] + tile[ty + 1][tx]);
    }
}

// Type 3: each block produces 16x16 outputs.
// Edge threads additionally load the one-pixel halo.
__global__ void shared_halo_stencil_kernel(float *in, float *out, int N)
{
    __shared__ float tile[IN_TILE_DIM + 2][IN_TILE_DIM + 2];
    int tx = threadIdx.x, ty = threadIdx.y;
    int col = blockIdx.x * IN_TILE_DIM + tx;
    int row = blockIdx.y * IN_TILE_DIM + ty;
    int x = tx + 1, y = ty + 1; // Leave room for halo cells.

    tile[y][x] = (row < N && col < N) ? in[row * N + col] : 0.0f;
    if (tx == 0)
        tile[y][0] = (row < N && col > 0) ? in[row * N + col - 1] : 0.0f;
    if (tx == IN_TILE_DIM - 1)
        tile[y][IN_TILE_DIM + 1] = (row < N && col + 1 < N)
            ? in[row * N + col + 1] : 0.0f;
    if (ty == 0)
        tile[0][x] = (row > 0 && col < N) ? in[(row - 1) * N + col] : 0.0f;
    if (ty == IN_TILE_DIM - 1)
        tile[IN_TILE_DIM + 1][x] = (row + 1 < N && col < N)
            ? in[(row + 1) * N + col] : 0.0f;

    __syncthreads(); // All center and halo values are ready.
    if (row > 0 && row < N - 1 && col > 0 && col < N - 1)
        out[row * N + col] = 0.2f * (
            tile[y][x] + tile[y][x - 1] + tile[y][x + 1] +
            tile[y - 1][x] + tile[y + 1][x]);
}
