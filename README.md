# CUDA stencil — compact shared-memory implementation

Two shared-memory kernels now live in **one file** (`src/shared_stencil.cu`).
The compiled CUDA program handles timing, correctness checks, and CSV output,
so no separate benchmarking script is needed.

## Files to add or replace

- `src/main.cu` — image I/O, kernel selection, CUDA-event timing, CPU check, CSV
- `src/shared_stencil.cu` — textbook shared tile (type 2) and halo tile (type 3)
- `src/stencils.cuh` — kernel declarations and tile sizes
- `Makefile` — build, correctness test, and all-image benchmark
- `README.md` — instructions

**Keep** the original `src/basic_stencil.cu` (type 1) and `data/` directory.
**Remove** obsolete `src/shared_halo_stencil.cu`, `tools/benchmark.sh`, and
`tools/benchmark.py`, if they exist. They are no longer used.

## Build and test on PSU Linux lab

```bash
nvidia-smi
nvcc --version
make ARCH=sm_86
make test
make benchmark REPEATS=100
```

Or run one image directly:

```bash
./stencil 512 data/words.txt 0 100     # all 3 kernels + CSV
./stencil 512 data/words.txt 3 100     # halo kernel only
```

`make benchmark` tests all three provided 512x512 images. Results go to
`result/benchmark.csv`; filtered images go into `result/<kernel>/`.
A 16x16 CUDA thread block is used for every kernel. Type 2 produces a 14x14
output tile; type 3 produces a 16x16 output tile using extra halo loads.

`ARCH=sm_86` supports RTX 3060 / RTX A2000. Confirm the actual GPU with
`nvidia-smi`; change `ARCH` if the lab GPU has a different compute capability.
Timing excludes disk I/O and CPU/GPU transfers. GPU testing is required.
