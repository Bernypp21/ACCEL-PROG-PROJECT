# CUDA five-point stencil: incremental update

Extends the previous **commented shared-memory starter**, following the simple
load / synchronize / compute pattern from *Programming Massively Parallel
Processors*, Chapter 8, Section 8.4.

## What changed (five files)

- `src/shared_halo_stencil.cu` — **new** third kernel; produces a full 16x16
  output tile using an 18x18 shared-memory array (one-pixel halo).
- `src/stencils.cuh` — declares the new kernel.
- `src/main.cu` — adds kernel type 3, GPU-event timing and CPU correctness checks.
- `Makefile` — simple build, test and clean commands.
- `README.md` — setup and explanations.

**Unchanged:** `src/basic_stencil.cu`, `src/shared_stencil.cu`, images and
`src/register_stencil.cu` (still empty).

## How the three versions differ

| Type | Kernel | What it does |
| --- | --- | --- |
| 1 | Basic | Reads all five pixels from global memory. |
| 2 | Shared starter | Loads a 16x16 tile; 14x14 threads produce outputs. |
| 3 | Shared halo | Loads a 16x16 tile **plus** halo pixels; all 16x16 threads can produce outputs (except image borders). |

For type 3, each thread loads its center pixel, the edge threads additionally
load nearby halo pixels, and `__syncthreads()` makes them available to the block.
The four corner halo cells are unnecessary for a five-point stencil.

All three kernels calculate:

    output = 0.2 * (center + left + right + up + down)

The outer image border stays zero, just as in the original project.

## Build on the PSU Linux lab

Log in to a CUDA-capable lab machine such as `cyclotron`, then run **from the
project directory**:

```bash
nvidia-smi
nvcc --version
make
```

`make` targets `sm_86`, appropriate for NVIDIA Ampere GPUs such as an RTX A2000
or RTX 3060. Verify the installed GPU on the actual lab host with `nvidia-smi`.
If the GPU uses another architecture, run `make ARCH=sm_XX` with its supported
compute capability. If `nvcc` is unavailable or rejects the system's GCC,
check the installed CUDA toolkit and supported host compiler.

## Run, validate and benchmark

```bash
mkdir -p result/basic_stencil result/shared_stencil result/shared_halo_stencil
./stencil 512 data/words.txt 1 100
./stencil 512 data/words.txt 2 100
./stencil 512 data/words.txt 3 100
```

The final number (`100`) is optional and repeats the kernel 100 times for timing.
Each run prints **average GPU kernel time** and a **PASS/FAIL correctness check**
against the CPU. It does not include disk I/O or host-device transfers.

To build and run all three automatically:

```bash
make test
```

`make test` compares the three saved files with `diff`. No differences means
all outputs match in the existing two-decimal text format.

You can also try `data/bush.txt` and `data/ceiling.txt` (both 512x512).
For other N sizes, the input file must begin with matching dimensions.

## Next steps

1. Record GPU times for types 1, 2 and 3 on the same image.
2. Calculate speedup: `basic time / optimized time`.
3. Repeat at different image sizes and explain the effect of shared-memory
   reuse, halo loads and synchronization.

**Testing note:** Native `nvcc` execution still needs to be done on the lab GPU;
this archive can be checked structurally and via CPU reference simulation in
an environment without CUDA.
