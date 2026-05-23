#include <iostream>
#include <math.h>

void add (int n, float* x, float* y) {
    for(int i=0; i<n; i++) {
        y[i] = x[i] + y[i];
    }
}

int main() {
    int N = 1<<20; //1milion

    float* x = new float[N];
    float* y = new float[N];

    //initialize x and y arrays on the host
    for(int i=0; i<N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    //Run kernel on 1M elements on the CPU
    add(N, x, y);

    //Check for errors
    float maxError = 0.0f;
    for(int i=0; i<N; i++) {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }

    std::cout << "Max error: " << maxError << "\n";

    delete [] x;
    delete [] y;

    return 0;
}