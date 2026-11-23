#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "stencils.cuh"

int load_input(const char *filename, float *input, int N);
int save_output(const char *filename, const float *output, int N);
int verify_output(const float *input, const float *output, int N);

// Report a CUDA error instead of silently continuing.
#define CUDA_CHECK(call) do {                                      \
    cudaError_t error = (call);                                    \
    if (error != cudaSuccess) {                                    \
        fprintf(stderr, "CUDA error: %s (%s:%d)\n",               \
                cudaGetErrorString(error), __FILE__, __LINE__);    \
        exit(EXIT_FAILURE);                                       \
    }                                                             \
} while (0)

// Keep all kernel choices in one place.
void launch_stencil(int type, dim3 blocks, dim3 threads,
                    float *d_in, float *d_out, int N)
{
    if (type == 1)
        parallel_stencil_kernel<<<blocks, threads>>>(d_in, d_out, N);
    else if (type == 2)
        shared_stencil_kernel<<<blocks, threads>>>(d_in, d_out, N);
    else
        shared_halo_stencil_kernel<<<blocks, threads>>>(d_in, d_out, N);
}

int main(int argc, char *argv[])
{
    // Example: ./stencil 512 data/words.txt 3 100
    if (argc != 4 && argc != 5)
    {
        printf("Usage: %s <N> <input_file> <type> [repeats]\n", argv[0]);
        printf("Types: 1 = basic, 2 = shared starter, 3 = shared halo\n");
        return EXIT_FAILURE;
    }

    int N = atoi(argv[1]);
    int type = atoi(argv[3]);
    int repeats = (argc == 5) ? atoi(argv[4]) : 1;
    if (N < 3 || type < 1 || type > 3 || repeats < 1)
    {
        fprintf(stderr, "N >= 3, type = 1..3, repeats >= 1 required.\n");
        return EXIT_FAILURE;
    }

    size_t count = (size_t)N * N;
    size_t bytes = count * sizeof(float);

    // Step 1: Read image pixels into CPU memory.
    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)calloc(count, sizeof(float));
    if (h_in == NULL || h_out == NULL)
    {
        fprintf(stderr, "Could not allocate CPU memory.\n");
        free(h_in);
        free(h_out);
        return EXIT_FAILURE;
    }
    if (!load_input(argv[2], h_in, N))
    {
        free(h_in);
        free(h_out);
        return EXIT_FAILURE;
    }

    // Step 2: Allocate GPU memory and transfer the input.
    float *d_in = NULL;
    float *d_out = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_in, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, bytes)); // Keep the image border at zero.

    dim3 threads(IN_TILE_DIM, IN_TILE_DIM);
    int tile_size = (type == 2) ? OUT_TILE_DIM : IN_TILE_DIM;
    dim3 blocks((N + tile_size - 1) / tile_size,
                (N + tile_size - 1) / tile_size);

    // Step 3: Warm up the GPU once before measuring time.
    launch_stencil(type, blocks, threads, d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 4: Time only the GPU kernel (not file reading or writing).
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; i++)
        launch_stencil(type, blocks, threads, d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Average kernel time: %.5f ms (%d runs)\n",
           milliseconds / repeats, repeats);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Step 5: Copy output to CPU and check against a simple CPU result.
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    int correct = verify_output(h_in, h_out, N);

    // Step 6: Save each kernel's result in its own folder.
    const char *output_file = (type == 1)
        ? "result/basic_stencil/stencil_output.txt"
        : (type == 2)
            ? "result/shared_stencil/stencil_output.txt"
            : "result/shared_halo_stencil/stencil_output.txt";
    int saved = save_output(output_file, h_out, N);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out);

    return (correct && saved) ? EXIT_SUCCESS : EXIT_FAILURE;
}

// Read a text image: width, height, then N*N pixel values.
int load_input(const char *filename, float *input, int N)
{
    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        fprintf(stderr, "Could not open input: %s\n", filename);
        return 0;
    }

    int width, height;
    if (fscanf(file, "%d %d", &width, &height) != 2 ||
        width != N || height != N)
    {
        fprintf(stderr, "Image size does not match N.\n");
        fclose(file);
        return 0;
    }

    for (size_t i = 0; i < (size_t)N * N; i++)
    {
        if (fscanf(file, "%f", &input[i]) != 1)
        {
            fprintf(stderr, "Invalid pixel at index %zu.\n", i);
            fclose(file);
            return 0;
        }
    }

    fclose(file);
    printf("Loaded %d x %d image.\n", N, N);
    return 1;
}

// Compare GPU output with the same five-point average computed on the CPU.
int verify_output(const float *input, const float *output, int N)
{
    int errors = 0;
    float max_error = 0.0f;

    for (int row = 0; row < N; row++)
    {
        for (int col = 0; col < N; col++)
        {
            int idx = row * N + col;
            float expected = 0.0f; // Basic kernel leaves border pixels at zero.

            if (row > 0 && row < N - 1 && col > 0 && col < N - 1)
                expected = 0.2f * (
                    input[idx] + input[idx - 1] + input[idx + 1] +
                    input[idx - N] + input[idx + N]);

            float difference = fabsf(output[idx] - expected);
            if (difference > max_error)
                max_error = difference;
            if (difference > 0.001f)
            {
                if (errors < 3)
                    printf("Mismatch at (%d, %d): GPU %.4f, CPU %.4f\n",
                           row, col, output[idx], expected);
                errors++;
            }
        }
    }

    printf("Correctness: %s (max error %.6f)\n",
           errors == 0 ? "PASS" : "FAIL", max_error);
    return errors == 0;
}

// Save the output in the original project's text format.
int save_output(const char *filename, const float *output, int N)
{
    FILE *file = fopen(filename, "w");
    if (file == NULL)
    {
        fprintf(stderr, "Could not write output: %s\n", filename);
        return 0;
    }

    fprintf(file, "%d %d\n", N, N);
    for (size_t i = 0; i < (size_t)N * N; i++)
    {
        fprintf(file, "%.2f ", output[i]);
        if ((i + 1) % N == 0)
            fprintf(file, "\n");
    }

    fclose(file);
    printf("Saved %s\n", filename);
    return 1;
}
