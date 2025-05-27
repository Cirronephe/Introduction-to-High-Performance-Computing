#include "spmm_opt.h"

const int BLOCK_SIZE = 128, THREAD_SIZE = 2;

__global__ void spmm_kernel_placeholder(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int INFEATURE)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int begin = tid * THREAD_SIZE, end = min((tid + 1) * THREAD_SIZE, ptr[num_v]);

    if (begin >= end) return;

    int l = 0, r = num_v;

    while (l < r) {
        int mid = (l + r + 1) >> 1;
        if (begin >= ptr[mid]) l = mid;
        else r = mid - 1;
    }    

    for (int j = 0; j < INFEATURE; ++j) {
        float result = 0.0f;
        int k = l;
        for (int i = begin; i < end; ++i) {
            if (i >= ptr[k + 1]) {
                atomicAdd(&vout[k * INFEATURE + j], result);
                result = 0.0f;
                while (i >= ptr[k + 1]) ++k;
            }
            result += vin[idx[i] * INFEATURE + j] * val[i];
        }
        atomicAdd(&vout[k * INFEATURE + j], result);
    }
}
void SpMMOpt::preprocess(float *vin, float *vout)
{
    grid.x = (num_e + BLOCK_SIZE * THREAD_SIZE - 1) / (BLOCK_SIZE * THREAD_SIZE);
    block.x = BLOCK_SIZE;
}

void SpMMOpt::run(float *vin, float *vout)
{
    spmm_kernel_placeholder<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
}