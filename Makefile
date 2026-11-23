# PSU Linux lab: Ampere GPUs use sm_86. Change ARCH if needed.
NVCC ?= nvcc
ARCH ?= sm_86
FLAGS = -O2 -std=c++17 -arch=$(ARCH)
SOURCES = src/main.cu src/basic_stencil.cu src/shared_stencil.cu src/shared_halo_stencil.cu

all: stencil

stencil: $(SOURCES) src/stencils.cuh
	$(NVCC) $(FLAGS) $(SOURCES) -o stencil

test: stencil
	mkdir -p result/basic_stencil result/shared_stencil result/shared_halo_stencil
	./stencil 512 data/words.txt 1 100
	./stencil 512 data/words.txt 2 100
	./stencil 512 data/words.txt 3 100
	diff result/basic_stencil/stencil_output.txt result/shared_stencil/stencil_output.txt
	diff result/basic_stencil/stencil_output.txt result/shared_halo_stencil/stencil_output.txt

clean:
	rm -f stencil

.PHONY: all test clean
