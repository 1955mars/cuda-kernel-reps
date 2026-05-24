#include <iostream>
#include <math.h>

__global__
void add (int n, float* x, float* y) {
    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int i = index; i < n; i += stride) {
        y[i] = x[i] + y[i];
    }
}

int main() {
    int N = 1<<20; //1milion

    float* x;
    float* y;
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    //initialize x and y arrays on the host
    for(int i=0; i<N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    //Run kernel on 1M elements on the GPU
    //<<<1, 256>>> 1 block of 256 threads
    add<<<1, 256>>>(N, x, y);

    // wait for the GPU to finish before reading y on the host
    cudaDeviceSynchronize();

    //Check for errors
    float maxError = 0.0f;
    for(int i=0; i<N; i++) {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }

    std::cout << "Max error: " << maxError << "\n";

    cudaFree(x);
    cudaFree(y);

    return 0;
}