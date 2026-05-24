#include <iostream>
#include <math.h>

#define TILE 16

__global__
void matmul(int N, float* A, float* B, float* C) {
    __shared__ float As[TILE][TILE]; //Per Block
    __shared__ float Bs[TILE][TILE]; //Per Block

    int col = blockIdx.x * blockDim.x + threadIdx.x; //Per Thread
    int row = blockIdx.y * blockDim.y + threadIdx.y; //Per Thread
    float val = 0.0f; // Per Thread

    for(int tile_k = 0; tile_k < N/TILE; tile_k++) {
        As[threadIdx.y][threadIdx.x] = A[row * N + tile_k * TILE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(tile_k * TILE + threadIdx.y) * N + col];
        __syncthreads();

        for(int k=0; k<TILE; k++) {
            val += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = val; // per Thread
    }
}

int main() {

    int N = 1024;
    float *A, *B, *C;

    cudaMallocManaged(&A, N*N*sizeof(float));
    cudaMallocManaged(&B, N*N*sizeof(float));
    cudaMallocManaged(&C, N*N*sizeof(float));

    for(int i=0; i<N*N; i++) {
        A[i] = 1.0f;
        B[i] = 1.0f;
        C[i] = 0.0f;
    }

    dim3 block(16, 16);
    dim3 grid((N+15)/16, (N+15)/16);

    matmul<<<grid, block>>> (N, A, B, C);

    cudaDeviceSynchronize();

    float maxError = 0.0f;
    for(int i=0; i<N*N; i++) {
        maxError = fmax(maxError, fabs(C[i] - 1024.0f));
    }

    std::cout << "Max Error: " << maxError << "\n";

    return 0;
}