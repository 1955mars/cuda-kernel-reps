# Top-level Makefile — each step builds independently.
#
# `make`              builds everything available on this machine
# `make 01_vector_add` builds only that step
# `make clean`        wipes binaries
#
# Steps 01 is pure C++ (g++/clang++).
# Steps 02 onward require nvcc (Linux + NVIDIA GPU).

CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra

.PHONY: all clean 01_vector_add

all: 01_vector_add

# ---------------------------------------------------------------------------
# Step 01 — CPU vector add baseline (pure C++)
# ---------------------------------------------------------------------------
01_vector_add: 01_vector_add/vec_add

01_vector_add/vec_add: 01_vector_add/add.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# ---------------------------------------------------------------------------
clean:
	rm -f 01_vector_add/vec_add
