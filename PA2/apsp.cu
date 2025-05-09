// PLEASE MODIFY THIS FILE TO IMPLEMENT YOUR SOLUTION

// Brute Force APSP Implementation:

#include "apsp.h"

const int K = 32, B = 64, T = 2;

template <typename T>
inline T ceiling(T x, T y) {
    return (x + y - 1) / y;
}

namespace {

__global__ void kernel_phase1(int p, int n, int *graph) {
    int ty = threadIdx.y * T;
    int tx = threadIdx.x * T;
    int i = p * B + ty;
    int j = p * B + tx;

    __shared__ int shared_block[B][B];

    #pragma unroll T
    for (int u = 0; u < T; ++u) {
        int pi = i + u;
        if (pi < n)
            #pragma unroll T
            for (int v = 0; v < T; ++v) {
                int pj = j + v;
                if (pj < n)
                    shared_block[ty + u][tx + v] = graph[pi * n + pj];
            }
    }
    __syncthreads();

    int m = min(n - p * B, B);

    #pragma unroll K
    for (int k = 0; k < m; ++k) {
        #pragma unroll T
        for (int u = 0; u < T; ++u)
            #pragma unroll T
            for (int v = 0; v < T; ++v)
                shared_block[ty + u][tx + v] = min(shared_block[ty + u][tx + v], shared_block[ty + u][k] + shared_block[k][tx + v]);
        __syncthreads();
    }

    #pragma unroll T
    for (int u = 0; u < T; ++u) {
        int pi = i + u;
        if (pi < n)    
            #pragma unroll T
            for (int v = 0; v < T; ++v) {
                int pj = j + v;
                if (pj < n)
                    graph[pi * n + pj] = shared_block[ty + u][tx + v];
            }
    }
}

__global__ void kernel_phase2_row(int p, int n, int *graph) {
    int ty = threadIdx.y * T;
    int tx = threadIdx.x * T;
    int i = p * B + ty;
    int j = blockIdx.x * blockDim.x * T + tx;
    
    if (blockIdx.x == p) return;

    __shared__ int shared_pivot[B][B];
    int reg_block[T][T];

    #pragma unroll T
    for (int u = 0; u < T; ++u) {
        int pi = p * B + ty + u;
        if (pi < n)
            #pragma unroll T
            for (int v = 0; v < T; ++v) {
                int pj = p * B + tx + v;
                if (pj < n)
                    shared_pivot[ty + u][tx + v] = graph[pi * n + pj];
            }
    }
    #pragma unroll T
    for (int u = 0; u < T; ++u) {
        int pi = i + u;
        if (pi < n)
            #pragma unroll T
            for (int v = 0; v < T; ++v) {
                int pj = j + v;
                if (pj < n)
                    reg_block[u][v] = graph[pi * n + pj];
            }
    }
    __syncthreads();

    int m = min(n - p * B, B);

    #pragma unroll K
    for (int k = 0; k < m; ++k) {
        #pragma unroll T
        for (int u = 0; u < T; ++u)
            #pragma unroll T
            for (int v = 0; v < T; ++v)
                if (j + v < n)
                    reg_block[u][v] = min(reg_block[u][v], shared_pivot[ty + u][k] + graph[(p * B + k) * n + j + v]);
    }
    
    #pragma unroll T
    for (int u = 0; u < T; ++u) {
        int pi = i + u;
        if (pi < n)
            #pragma unroll T
            for (int v = 0; v < T; ++v) {
                int pj = j + v;
                if (pj < n)
                    graph[pi * n + pj] = reg_block[u][v];
            }
    }
}

__global__ void kernel_phase2_col(int p, int n, int *graph) {
    int ty = threadIdx.y * T;
    int tx = threadIdx.x * T;
    int i = blockIdx.y * blockDim.y * T + ty;
    int j = p * B + tx;
    
    if (blockIdx.y == p) return;

    __shared__ int shared_pivot[B][B];
    int reg_block[T][T];

    #pragma unroll T
    for (int u = 0; u < T; ++u) {
        int pi = p * B + ty + u;
        if (pi < n)
            #pragma unroll T
            for (int v = 0; v < T; ++v) {
                int pj = p * B + tx + v;
                if (pj < n)
                    shared_pivot[ty + u][tx + v] = graph[pi * n + pj];
            }
    }
    #pragma unroll T
    for (int u = 0; u < T; ++u) {
        int pi = i + u;
        if (pi < n)
            #pragma unroll T
            for (int v = 0; v < T; ++v) {
                int pj = j + v;
                if (pj < n)
                    reg_block[u][v] = graph[pi * n + pj];
            }
    }
    __syncthreads();

    int m = min(n - p * B, B);

    #pragma unroll K
    for (int k = 0; k < m; ++k) {
        #pragma unroll T
        for (int u = 0; u < T; ++u)
            if (i + u < n)
                #pragma unroll T
                for (int v = 0; v < T; ++v)
                    reg_block[u][v] = min(reg_block[u][v], graph[(i + u) * n + p * B + k] + shared_pivot[k][tx + v]);
    }
    
    #pragma unroll T
    for (int u = 0; u < T; ++u) {
        int pi = i + u;
        if (pi < n)
            #pragma unroll T
            for (int v = 0; v < T; ++v) {
                int pj = j + v;
                if (pj < n)
                    graph[pi * n + pj] = reg_block[u][v];
            }
    }
}

__global__ void kernel_phase3(int p, int n, int *graph) {
    int ty = threadIdx.y * T;
    int tx = threadIdx.x * T;
    int i = blockIdx.y * blockDim.y * T + ty;
    int j = blockIdx.x * blockDim.x * T + tx;

    if (blockIdx.y == p) return;
    if (blockIdx.x == p) return;

    __shared__ int shared_pivot_row[B][B], shared_pivot_col[B][B];
    int reg_block[T][T];
    
    #pragma unroll T
    for (int u = 0; u < T; ++u) {
        int pi = i + u;
        if (pi < n)
            #pragma unroll T
            for (int v = 0; v < T; ++v) {
                int pj = p * B + tx + v;
                if (pj < n)
                    shared_pivot_row[ty + u][tx + v] = graph[pi * n + pj];
            }
    }
    #pragma unroll T
    for (int u = 0; u < T; ++u) {
        int pi = p * B + ty + u;
        if (pi < n)
            #pragma unroll T
            for (int v = 0; v < T; ++v) {
                int pj = j + v;
                if (pj < n)
                    shared_pivot_col[ty + u][tx + v] = graph[pi * n + pj];
            }
    }    
    __syncthreads();

    if (i >= n || j >= n) return;
    
    #pragma unroll T
    for (int u = 0; u < T; ++u) {
        int pi = i + u;
        if (pi < n)
            #pragma unroll T
            for (int v = 0; v < T; ++v) {
                int pj = j + v;
                if (pj < n)
                    reg_block[u][v] = graph[pi * n + pj];
            }
    }
    
    int m = min(n - p * B, B);

    #pragma unroll K
    for (int k = 0; k < m; ++k)
        #pragma unroll T
        for (int u = 0; u < T; ++u)
            #pragma unroll T
            for (int v = 0; v < T; ++v)
                reg_block[u][v] = min(reg_block[u][v], shared_pivot_row[ty + u][k] + shared_pivot_col[k][tx + v]);
    
    #pragma unroll T
    for (int u = 0; u < T; ++u) {
        int pi = i + u;
        if (pi < n)
            #pragma unroll T
            for (int v = 0; v < T; ++v) {
                int pj = j + v;
                if (pj < n)
                    graph[pi * n + pj] = reg_block[u][v];
            }
    }
}

}

void apsp(int n, /* device */ int *graph) {
    int m = ceiling(n, B);
    dim3 thr(B / T, B / T);

    for (int p = 0; p < m; ++p) {
        kernel_phase1<<<dim3(1, 1), thr>>>(p, n, graph);
        kernel_phase2_row<<<dim3(m, 1), thr>>>(p, n, graph);
        kernel_phase2_col<<<dim3(1, m), thr>>>(p, n, graph);
        kernel_phase3<<<dim3(m, m), thr>>>(p, n, graph);
    }
}

