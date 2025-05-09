// PLEASE MODIFY THIS FILE TO IMPLEMENT YOUR SOLUTION

// Brute Force APSP Implementation:

#include "apsp.h"
#include <cstdio>

const int B = 64, TD = 2;

template <typename T>
inline T ceiling(T x, T y) {
    return (x + y - 1) / y;
}

namespace {

__global__ void kernel_phase1(int p, int b, int n, int *graph) {
    int ty = threadIdx.y * TD;
    int tx = threadIdx.x * TD;
    int i = p * b + ty;
    int j = p * b + tx;

    __shared__ int shared_block[B][B];

    for (int u = 0; u < TD; ++u) {
        int pi = i + u;
        if (pi < n)
            for (int v = 0; v < TD; ++v) {
                int pj = j + v;
                if (pj < n)
                    shared_block[ty + u][tx + v] = graph[pi * n + pj];
            }
    }
    __syncthreads();

    int m = min(n - p * b, b);
    #pragma unroll
    for (int k = 0; k < m; ++k) {
        for (int u = 0; u < TD; ++u)
            for (int v = 0; v < TD; ++v)
                shared_block[ty + u][tx + v] = min(shared_block[ty + u][tx + v], shared_block[ty + u][k] + shared_block[k][tx + v]);
        __syncthreads();
    }

    for (int u = 0; u < TD; ++u) {
        int pi = i + u;
        if (pi < n)    
            for (int v = 0; v < TD; ++v) {
                int pj = j + v;
                if (pj < n)
                    graph[pi * n + pj] = shared_block[ty + u][tx + v];
            }
    }
}

__global__ void kernel_phase2_row(int p, int b, int n, int *graph) {
    int ty = threadIdx.y * TD;
    int tx = threadIdx.x * TD;
    int i = p * b + ty;
    int j = blockIdx.x * blockDim.x * TD + tx;
    
    if (blockIdx.x == p) return;

    __shared__ int shared_pivot[B][B];
    int reg_block[TD][TD];

    for (int u = 0; u < TD; ++u) {
        int pi = p * b + ty + u;
        if (pi < n)
            for (int v = 0; v < TD; ++v) {
                int pj = p * b + tx + v;
                if (pj < n)
                    shared_pivot[ty + u][tx + v] = graph[pi * n + pj];
            }
    }
    for (int u = 0; u < TD; ++u) {
        int pi = i + u;
        if (pi < n)
            for (int v = 0; v < TD; ++v) {
                int pj = j + v;
                if (pj < n)
                    reg_block[u][v] = graph[pi * n + pj];
            }
    }
    __syncthreads();

    int m = min(n - p * b, b);
    #pragma unroll
    for (int k = 0; k < m; ++k) {
        for (int u = 0; u < TD; ++u)
            for (int v = 0; v < TD; ++v)
                if (j + v < n)
                    reg_block[u][v] = min(reg_block[u][v], shared_pivot[ty + u][k] + graph[(p * b + k) * n + j + v]);
    }
    
    for (int u = 0; u < TD; ++u) {
        int pi = i + u;
        if (pi < n)
            for (int v = 0; v < TD; ++v) {
                int pj = j + v;
                if (pj < n)
                    graph[pi * n + pj] = reg_block[u][v];
            }
    }
}

__global__ void kernel_phase2_col(int p, int b, int n, int *graph) {
    int ty = threadIdx.y * TD;
    int tx = threadIdx.x * TD;
    int i = blockIdx.y * blockDim.y * TD + ty;
    int j = p * b + tx;
    
    if (blockIdx.y == p) return;

    __shared__ int shared_pivot[B][B];
    int reg_block[TD][TD];

    for (int u = 0; u < TD; ++u) {
        int pi = p * b + ty + u;
        if (pi < n)
            for (int v = 0; v < TD; ++v) {
                int pj = p * b + tx + v;
                if (pj < n)
                    shared_pivot[ty + u][tx + v] = graph[pi * n + pj];
            }
    }
    for (int u = 0; u < TD; ++u) {
        int pi = i + u;
        if (pi < n)
            for (int v = 0; v < TD; ++v) {
                int pj = j + v;
                if (pj < n)
                    reg_block[u][v] = graph[pi * n + pj];
            }
    }
    __syncthreads();

    int m = min(n - p * b, b);
    #pragma unroll
    for (int k = 0; k < m; ++k) {
        for (int u = 0; u < TD; ++u)
            if (i + u < n)
                for (int v = 0; v < TD; ++v)
                    reg_block[u][v] = min(reg_block[u][v], graph[(i + u) * n + p * b + k] + shared_pivot[k][tx + v]);
    }
    
    for (int u = 0; u < TD; ++u) {
        int pi = i + u;
        if (pi < n)
            for (int v = 0; v < TD; ++v) {
                int pj = j + v;
                if (pj < n)
                    graph[pi * n + pj] = reg_block[u][v];
            }
    }
}

__global__ void kernel_phase3(int p, int b, int n, int *graph) {
    int ty = threadIdx.y * TD;
    int tx = threadIdx.x * TD;
    int i = blockIdx.y * blockDim.y * TD + ty;
    int j = blockIdx.x * blockDim.x * TD + tx;
    
    if (blockIdx.y == p) return;
    if (blockIdx.x == p) return;

    __shared__ int shared_pivot_row[B][B], shared_pivot_col[B][B];
    int reg_block[TD][TD];
    
    for (int u = 0; u < TD; ++u) {
        int pi = i + u;
        if (pi < n)
            for (int v = 0; v < TD; ++v) {
                int pj = p * b + tx + v;
                if (pj < n)
                    shared_pivot_row[ty + u][tx + v] = graph[pi * n + pj];
            }
    }
    for (int u = 0; u < TD; ++u) {
        int pi = p * b + ty + u;
        if (pi < n)
            for (int v = 0; v < TD; ++v) {
                int pj = j + v;
                if (pj < n)
                    shared_pivot_col[ty + u][tx + v] = graph[pi * n + pj];
            }
    }    
    for (int u = 0; u < TD; ++u) {
        int pi = i + u;
        if (pi < n)
            for (int v = 0; v < TD; ++v) {
                int pj = j + v;
                if (pj < n)
                    reg_block[u][v] = graph[pi * n + pj];
            }
    }
    __syncthreads();
    
    int m = min(n - p * b, b);
    #pragma unroll
    for (int k = 0; k < m; ++k) {
        for (int u = 0; u < TD; ++u)
            for (int v = 0; v < TD; ++v)
                reg_block[u][v] = min(reg_block[u][v], shared_pivot_row[ty + u][k] + shared_pivot_col[k][tx + v]);
    }
    
    for (int u = 0; u < TD; ++u) {
        int pi = i + u;
        if (pi < n)
            for (int v = 0; v < TD; ++v) {
                int pj = j + v;
                if (pj < n)
                    graph[pi * n + pj] = reg_block[u][v];
            }
    }
}

}

void apsp(int n, /* device */ int *graph) {
    int b = B, m = ceiling(n, b);
    dim3 thr(B / TD, B / TD);
    for (int p = 0; p < m; ++p) {
        kernel_phase1<<<dim3(1, 1), thr>>>(p, b, n, graph);
        kernel_phase2_row<<<dim3(m, 1), thr>>>(p, b, n, graph);
        kernel_phase2_col<<<dim3(1, m), thr>>>(p, b, n, graph);
        kernel_phase3<<<dim3(m, m), thr>>>(p, b, n, graph);
    }
}

