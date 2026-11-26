# Compile for the GPU installed on the lab machine (default: sm_86).
NVCC ?= nvcc
ARCH ?= sm_86
REPEATS ?= 100
N ?= 512
IMAGES ?= data/words.txt data/bush.txt data/ceiling.txt
SOURCES = src/main.cu src/basic_stencil.cu src/shared_stencil.cu

stencil: $(SOURCES) src/stencils.cuh
	$(NVCC) -O2 -std=c++17 -arch=$(ARCH) $(SOURCES) -o $@

test: stencil
	./stencil $(N) data/words.txt 0 10

benchmark: stencil
	@mkdir -p result
	@rm -f result/benchmark.csv
	@for image in $(IMAGES); do ./stencil $(N) "$$image" 0 $(REPEATS) || exit 1; done

clean:
	rm -f stencil

.PHONY: test benchmark clean
