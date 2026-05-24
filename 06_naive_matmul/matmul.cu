#include <iostream>
#include <math.h>

__global__
void matmul(int N, float* A, float* B, float* C) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < N && col < N) {
        float val = 0.0f;
        for(int k=0; k<N; k++) {
            val += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = val;
    }
}

int main() {

    int N = 1024; //N*N Matrix

    float* A;
    float* B;
    float* C;

    cudaMallocManaged(&A, N*N*sizeof(float));
    cudaMallocManaged(&B, N*N*sizeof(float));
    cudaMallocManaged(&C, N*N*sizeof(float));

    for(int i=0; i<N*N; i++) {
        A[i] = 1.0f;
        B[i] = 1.0f;
        C[i] = 0.0f;
    }

    dim3 block(16, 16); //256 threads in 16x16 table
    dim3 grid((N+15)/16, (N+15)/16); 

    matmul<<<grid, block>>>(N, A, B, C);

    cudaDeviceSynchronize();

    float maxError = 0.0f;
    for(int i=0; i<N*N; i++) {
        maxError = fmax(maxError, fabs(C[i] - 1024.0f));
    }

    std::cout << "Max error: " << maxError << "\n";

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    

    return 0;
}