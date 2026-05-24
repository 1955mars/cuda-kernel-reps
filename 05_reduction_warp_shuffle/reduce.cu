#include <iostream>
#include <math.h>

__global__
void reduce(int n, float* x, float* out) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float val = index < n ? x[index] : 0.0f;


    //Stage 1 - Warp Shuffle Reduction (32 -> 1)
    for(int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    //Stage 2 - Shared Memory (8 warps -> 1 block)
    __shared__ float warp_sums[8];
    int lane = threadIdx.x & 31; //which lane within a warp (0-31)
    int warp_id = threadIdx.x >> 5; //division by 32, which warp in a block (0-7) (256 thread or 8 blocks)
    if(lane == 0) warp_sums[warp_id] = val;
    __syncthreads();

    //Stage 3 - thread 0 finishes the block, then atomicAdd
    if (threadIdx.x == 0) {
        float block_sum = 0.0f;
        for(int w=0; w<8; w++) {
            block_sum += warp_sums[w];
        }
        //Thread 0's block_sum register of all 4096 blocks have each blocks sum
        //Below all blocks write to out in atomic fashion
        atomicAdd(out, block_sum);
    }
}

int main() {
    int N = 1<<20; // 1 Million

    float* x;
    float* out;

    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&out, sizeof(float));

    for(int i=0; i<N; i++) x[i] = 1.0f;
    *out = 0.0f;

    int B = 256; //threads per block
    int G = (N + B - 1)/B; //blocks per grid

    reduce<<<G, B>>>(N, x, out);
    cudaDeviceSynchronize();

    std::cout << "Sum: " << *out << " Expected: " << (float)N << " Error:" << fabs(*out - (float)N) << "\n";

    cudaFree(x);
    cudaFree(out);

    return 0;


}