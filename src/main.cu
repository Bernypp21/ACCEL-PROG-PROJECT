#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <sys/stat.h>
#include "stencils.cuh"

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                       \
    }                                                             \
} while (0)

// Read a square image stored as: width height, followed by pixels.
bool load_image(const char *path, std::vector<float> &pixels, int N)
{
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); return false; }
    int width, height;
    bool ok = fscanf(f, "%d %d", &width, &height) == 2 &&
              width == N && height == N;
    for (size_t i = 0; ok && i < pixels.size(); ++i)
        ok = fscanf(f, "%f", &pixels[i]) == 1;
    fclose(f);
    if (!ok) fprintf(stderr, "Invalid image or dimensions: %s\n", path);
    return ok;
}

// Compare GPU pixels with the same five-point average on the CPU.
bool verify(const std::vector<float> &in, const std::vector<float> &out, int N)
{
    for (int row = 0; row < N; ++row)
        for (int col = 0; col < N; ++col) {
            int p = row * N + col;
            float expected = 0.0f; // Keep image borders zero.
            if (row > 0 && row < N-1 && col > 0 && col < N-1)
                expected = 0.2f * (in[p] + in[p-1] + in[p+1] +
                                   in[p-N] + in[p+N]);
            if (!(std::fabs(out[p] - expected) <= 0.001f)) {
                printf("Mismatch at row %d, col %d\n", row, col);
                return false;
            }
        }
    return true;
}

bool save_image(const char *folder, const std::vector<float> &out, int N)
{
    mkdir("result", 0755);
    mkdir(folder, 0755);
    char path[160];
    snprintf(path, sizeof(path), "%s/stencil_output.txt", folder);
    FILE *f = fopen(path, "w");
    if (!f) { perror(path); return false; }
    fprintf(f, "%d %d\n", N, N);
    for (size_t i = 0; i < out.size(); ++i)
        fprintf(f, "%.2f%c", out[i], (i + 1) % N ? ' ' : '\n');
    fclose(f);
    return true;
}

void launch(int type, dim3 grid, dim3 block, float *in, float *out, int N)
{
    if (type == 1) parallel_stencil_kernel<<<grid, block>>>(in, out, N);
    if (type == 2) shared_stencil_kernel<<<grid, block>>>(in, out, N);
    if (type == 3) shared_halo_stencil_kernel<<<grid, block>>>(in, out, N);
}

int main(int argc, char **argv)
{
    if (argc < 4 || argc > 5) {
        fprintf(stderr, "Usage: %s N image.txt type [repeats]\n"
                        "Types: 0=benchmark all, 1=basic, 2=shared, 3=shared halo\n", argv[0]);
        return 1;
    }
    int N = atoi(argv[1]), type = atoi(argv[3]);
    int repeats = argc == 5 ? atoi(argv[4]) : 100;
    if (N < 3 || type < 0 || type > 3 || repeats < 1) {
        fprintf(stderr, "Expected N>=3, type=0..3, repeats>=1.\n");
        return 1;
    }

    const char *names[] = {"", "Basic", "Shared", "SharedHalo"};
    const char *folders[] = {"", "result/basic_stencil",
                             "result/shared_stencil", "result/shared_halo_stencil"};
    size_t count = (size_t)N * N, bytes = count * sizeof(float);
    std::vector<float> input(count), output(count), baseline(count);
    if (!load_image(argv[2], input, N)) return 1;

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_in, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, input.data(), bytes, cudaMemcpyHostToDevice));

    float base_ms = 0.0f;
    bool ok = true;
    for (int t = type == 0 ? 1 : type; t <= (type == 0 ? 3 : type); ++t) {
        // Type 2 outputs 14x14 pixels per block; other types output 16x16.
        int step = t == 2 ? OUT_TILE_DIM : IN_TILE_DIM;
        dim3 block(IN_TILE_DIM, IN_TILE_DIM);
        dim3 grid((N + step - 1) / step, (N + step - 1) / step);
        CUDA_CHECK(cudaMemset(d_out, 0, bytes));
        launch(t, grid, block, d_in, d_out, N); // Warm up GPU.
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < repeats; ++i)
            launch(t, grid, block, d_in, d_out, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float total_ms;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        float ms = total_ms / repeats;

        CUDA_CHECK(cudaMemcpy(output.data(), d_out, bytes, cudaMemcpyDeviceToHost));
        bool correct = verify(input, output, N);
        bool same = true;
        if (t == 1) {
            baseline = output;
            base_ms = ms;
        } else if (type == 0) {
            for (size_t i = 0; i < count; ++i)
                if (!(std::fabs(output[i] - baseline[i]) <= 0.001f)) {
                    same = false;
                    break;
                }
        }
        bool saved = save_image(folders[t], output, N);
        printf("%-10s Average kernel time: %.6f ms | CPU: %s | Baseline: %s\n",
               names[t], ms, correct ? "PASS" : "FAIL", same ? "PASS" : "FAIL");
        ok &= correct && same && saved;

        // The benchmark writes a CSV; Makefile clears it before a new run.
        if (type == 0) {
            mkdir("result", 0755);
            FILE *csv = fopen("result/benchmark.csv", "a+");
            if (!csv) { perror("result/benchmark.csv"); ok = false; }
            else {
                fseek(csv, 0, SEEK_END);
                if (ftell(csv) == 0)
                    fprintf(csv, "image,size,kernel,average_ms,speedup_vs_basic,matches_basic\n");
                fprintf(csv, "%s,%d,%s,%.6f,%.3f,%s\n", argv[2], N, names[t],
                        ms, ms > 0.0f ? base_ms / ms : 0.0f, same ? "PASS" : "FAIL");
                fclose(csv);
            }
        }
        if (!ok) break;
    }
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return ok ? 0 : 1;
}
