// PLEASE MODIFY THIS FILE TO IMPLEMENT YOUR SOLUTION

// Brute Force APSP Implementation:

#include "apsp.h"
#include <cstdio>

template <typename T>
inline T ceiling(T x, T y) {
    return (x + y - 1) / y;
}

namespace {

__global__ void kernel_phase1(int p, int b, int n, int *graph) {
    const int TD = 1;
    int ty = threadIdx.y * TD;
    int tx = threadIdx.x * TD;
    int i = p * b + ty;
    int j = p * b + tx;

    __shared__ int shared_block[32][32];

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
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int i = p * b + ty;
    int j = blockIdx.x * blockDim.x + tx;
    
    if (j >= p * b) j += b;

    __shared__ int shared_block[32][32], shared_pivot[32][32];

    int pi = p * b + ty;
    int pj = p * b + tx;

    if (pi < n && pj < n) {
        shared_pivot[ty][tx] = graph[pi * n + pj];
    }
    if (i < n && j < n) {
        shared_block[ty][tx] = graph[i * n + j];
    }
    __syncthreads();

    int m = min(n - p * b, b);
    for (int k = 0; k < m; ++k) {
        shared_block[ty][tx] = min(shared_block[ty][tx], shared_pivot[ty][k] + shared_block[k][tx]);
        __syncthreads();
    }
    
    if (i < n && j < n) {
        graph[i * n + j] = shared_block[ty][tx];
    }
}

__global__ void kernel_phase2_col(int p, int b, int n, int *graph) {
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int i = blockIdx.y * blockDim.y + ty;
    int j = p * b + tx;
    
    if (i >= p * b) i += b;

    __shared__ int shared_block[32][32], shared_pivot[32][32];

    int pi = p * b + ty;
    int pj = p * b + tx;

    if (pi < n && pj < n) {
        shared_pivot[ty][tx] = graph[pi * n + pj];
    }
    if (i < n && j < n) {
        shared_block[ty][tx] = graph[i * n + j];
    }
    __syncthreads();

    int m = min(n - p * b, b);
    for (int k = 0; k < m; ++k) {
        shared_block[ty][tx] = min(shared_block[ty][tx], shared_block[ty][k] + shared_pivot[k][tx]);
        __syncthreads();
    }
    
    if (i < n && j < n) {
        graph[i * n + j] = shared_block[ty][tx];
    }
}

__global__ void kernel_phase3(int p, int b, int n, int *graph) {
    const int TD = 2;
    int ty = threadIdx.y * TD;
    int tx = threadIdx.x * TD;
    int i = blockIdx.y * blockDim.y * TD + ty;
    int j = blockIdx.x * blockDim.x * TD + tx;
    
    if (i >= p * b) i += b;
    if (j >= p * b) j += b;

    __shared__ int shared_block[32][32], shared_pivot_row[32][32], shared_pivot_col[32][32];
    
    #pragma unroll
    for (int u = 0; u < TD; ++u) {
        int pi = i + u;
        if (pi < n)
            #pragma unroll
            for (int v = 0; v < TD; ++v) {
                int pj = p * b + tx + v;
                if (pj < n)
                    shared_pivot_row[ty + u][tx + v] = graph[pi * n + pj];
            }
    }
    #pragma unroll
    for (int u = 0; u < TD; ++u) {
        int pi = p * b + ty + u;
        if (pi < n)
            #pragma unroll
            for (int v = 0; v < TD; ++v) {
                int pj = j + v;
                if (pj < n)
                    shared_pivot_col[ty + u][tx + v] = graph[pi * n + pj];
            }
    }
    #pragma unroll
    for (int u = 0; u < TD; ++u) {
        int pi = i + u;
        if (pi < n)
            #pragma unroll
            for (int v = 0; v < TD; ++v) {
                int pj = j + v;
                if (pj < n)
                    shared_block[ty + u][tx + v] = graph[pi * n + pj];
            }
    }
    __syncthreads();

    int reg_block[TD][TD];

    int m = min(n - p * b, b);
    #pragma unroll
    for (int u = 0; u < TD; ++u)
        #pragma unroll
        for (int v = 0; v < TD; ++v)
            reg_block[u][v] = shared_block[ty + u][tx + v];

    #pragma unroll
    for (int k = 0; k < m; ++k) {
        #pragma unroll
        for (int u = 0; u < TD; ++u)
            #pragma unroll
            for (int v = 0; v < TD; ++v)
                reg_block[u][v] = min(reg_block[u][v], shared_pivot_row[ty + u][k] + shared_pivot_col[k][tx + v]);
    }
    
    #pragma unroll
    for (int u = 0; u < TD; ++u) {
        int pi = i + u;
        if (pi < n)
            #pragma unroll
            for (int v = 0; v < TD; ++v) {
                int pj = j + v;
                if (pj < n)
                    graph[pi * n + pj] = reg_block[u][v];
            }
    }
}

}

void apsp(int n, /* device */ int *graph) {
    int b = 32, m = ceiling(n, b);
    for (int p = 0; p < m; ++p) {
        dim3 thr(32, 32);
        kernel_phase1<<<dim3(1, 1), dim3(32, 32)>>>(p, b, n, graph);
        kernel_phase2_row<<<dim3(m - 1, 1), thr>>>(p, b, n, graph);
        kernel_phase2_col<<<dim3(1, m - 1), thr>>>(p, b, n, graph);
        kernel_phase3<<<dim3(m - 1, m - 1), dim3(16, 16)>>>(p, b, n, graph);
    }
}

