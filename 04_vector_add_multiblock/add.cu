#include <iostream>
#include <math.h>

__global__
void add(int n, float* x, float* y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = index; i < n; i += stride) {
        y[i] = x[i] + y[i];
    }
 }
int main() {
    int N = 1<<20; //1 Million

    float* x;
    float* y;
    
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    //initialize x and y arrays on the host
    for(int i=0; i<N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
     }

    int numThreadsPerBlock = 256; // a good default 
    int numBlocksPerGrid = (N + numThreadsPerBlock - 1)/numThreadsPerBlock;

    add<<<numBlocksPerGrid, numThreadsPerBlock>>>(N, x, y);

    cudaDeviceSynchronize();

    float maxError = 0.0f;
    for(int i=0; i<N; i++) {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }

    std::cout << "Max error: " << maxError << "\n";

    cudaFree(x);
    cudaFree(y);

    return 0;


}