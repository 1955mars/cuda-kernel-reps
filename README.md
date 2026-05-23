# cuda-kernel-reps

Hand-written GPU kernels — building CUDA + Triton intuition from
SAXPY through tiled matmul, following Mark Harris's NVIDIA intro
and Simon Boehm's matmul writeup.

Companion to [tiny-autodiff](https://github.com/1955mars/tiny-autodiff) —
that repo built the substrate above the GPU (DAG construction + backward
traversal); this one builds the substrate below it (how a single op
runs fast on parallel hardware).

## Status

- [ ] 01 — CPU vector add (baseline)
- [ ] 02 — Naive CUDA vector add (`<<<1, 1>>>`)
- [ ] 03 — Single-block parallel vector add (`<<<1, N>>>`)
- [ ] 04 — Multi-block vector add (`<<<G, B>>>`)
- [ ] 05 — Reduction with warp shuffles
- [ ] 06 — Naive matmul
- [ ] 07 — Tiled matmul with shared memory (target: ≥ 50% of cuBLAS)
- [ ] 08 — cuBLAS comparison + Nsight Compute profile
- [ ] 09 — Triton SAXPY
- [ ] 10 — Triton fused softmax

## Build

```bash
make 01_vector_add
./01_vector_add/vec_add
```

Steps 01 is pure C++ (compiles with `g++` / `clang++` on macOS or Linux).
Steps 02 onward require `nvcc` from the CUDA Toolkit on a Linux box with
an NVIDIA GPU.

## Why

GPU kernel work is the foundational skill for rung-6 of the [A2B map](https://github.com/1955mars/tiny-autodiff)
(HPC + ML systems). Writing the kernels by hand — naive first, then
iteratively optimising while reading the profiler — is the only way to
internalise the warp / block / grid model, the memory hierarchy, and
the bandwidth-vs-compute trade-offs that production ML systems live or
die by.

## License

MIT — see [LICENSE](LICENSE).
