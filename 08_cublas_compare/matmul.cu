#include <iostream>
#include <math.h>
#include <cublas_v2.h>

#define TILE 16

__global__
void matmul(int N, float* A, float* B, float* C) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    float val = 0.0f;

    for(int tile_k = 0; tile_k < N/TILE; tile_k++){
        As[threadIdx.y][threadIdx.x] = A[row * N + tile_k * TILE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(tile_k * TILE + threadIdx.y) * N + col];
        __syncthreads();

        for(int k=0; k<TILE; k++) {
            val += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = val;

}

int main() {
    int N = 1024;

    float *A, *B, *C_Tiled, *C_Cublas;

    cudaMallocManaged(&A, N*N*sizeof(float));
    cudaMallocManaged(&B, N*N*sizeof(float));
    cudaMallocManaged(&C_Tiled, N*N*sizeof(float));
    cudaMallocManaged(&C_Cublas, N*N*sizeof(float));

    for(int i=0; i<N*N; i++) {
        A[i] = 1.0f;
        B[i] = 1.0f;
        C_Tiled[i] = 0.0f;
        C_Cublas[i] = 0.0f;
    }

    dim3 block(TILE, TILE);
    dim3 grid((N+TILE-1)/TILE, (N+TILE-1)/TILE);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    //warmup - first launch pays driver/JIT cost
    matmul<<<grid, block>>>(N, A, B, C_Tiled);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
    N, N, N, &alpha, B, N, A, N, &beta, C_Cublas, N);

    cudaDeviceSynchronize();

    //reset outputs
    for(int i=0; i<N*N; i++) {
        C_Tiled[i] = 0.0f;
        C_Cublas[i] = 0.0f;
    }

    //time tiled kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul<<<grid, block>>>(N, A, B, C_Tiled);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float tiled_ms = 0.0f;
    cudaEventElapsedTime(&tiled_ms, start, stop);

    //time cuBlas
    cudaEventRecord(start);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
    N, N, N, &alpha, B, N, A, N, &beta, C_Cublas, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float cublas_ms = 0.0f;
    cudaEventElapsedTime(&cublas_ms, start, stop);

    //GFLOPS
    float flops = 2.0f * N * N * N;
    float tiled_gflops = flops/(tiled_ms * 1e6f);
    float cublas_gflops = flops/(cublas_ms * 1e6f);

    //--verify---
    float maxErr_tiled = 0.0f, maxErr_cublas = 0.0f;
    for(int i=0; i<N*N; i++) {
        maxErr_tiled = fmax(maxErr_tiled, fabs(C_Tiled[i] - (float)N));
        maxErr_cublas = fmax(maxErr_cublas, fabs(C_Cublas[i] - (float)N));
    }

    std::cout << "TILED: " << "time " << tiled_ms << ", GFlops " << tiled_gflops << ", maxError " << maxErr_tiled << "\n";
    std::cout << "CUBLAS: " << "time " << cublas_ms << ", GFlops " << cublas_gflops << ", maxError " << maxErr_cublas << "\n";

    std::cout << "CUBLAS is " << tiled_gflops / cublas_gflops * 100.0f <<  "% matched by tiled \n";

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cublasDestroy(handle);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C_Tiled);
    cudaFree(C_Cublas);

    return 0;
}