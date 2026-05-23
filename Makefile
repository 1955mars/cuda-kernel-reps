# Top-level Makefile — each step builds independently.
#
# `make`                       builds everything available on this machine
# `make 01_vector_add`         builds only that step
# `make 02_vector_add_naive`   builds the CUDA step (needs nvcc)
# `make clean`                 wipes binaries
#
# Step 01 is pure C++ (g++/clang++).
# Step 02 onward require nvcc (Linux + NVIDIA GPU).

CXX        := g++
CXXFLAGS   := -std=c++17 -O2 -Wall -Wextra

NVCC       := nvcc
NVCC_FLAGS := -std=c++17 -O2

.PHONY: all clean 01_vector_add 02_vector_add_naive

all: 01_vector_add

# ---------------------------------------------------------------------------
# Step 01 — CPU vector add baseline (pure C++)
# ---------------------------------------------------------------------------
01_vector_add: 01_vector_add/vec_add

01_vector_add/vec_add: 01_vector_add/add.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# ---------------------------------------------------------------------------
# Step 02 — Naive CUDA vector add (<<<1, 1>>>)
# ---------------------------------------------------------------------------
02_vector_add_naive: 02_vector_add_naive/vec_add

02_vector_add_naive/vec_add: 02_vector_add_naive/add.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# ---------------------------------------------------------------------------
clean:
	rm -f 01_vector_add/vec_add 02_vector_add_naive/vec_add
