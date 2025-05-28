#include "spmm_opt.h"

const int BLOCK_SIZE = 128, WARP_SIZE = 32;

__global__ void spmm_kernel(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int feat_in, int THREAD_SIZE)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id >= num_v) return;

    int row_begin = ptr[warp_id], row_end = ptr[warp_id + 1];
    int feat_begin = THREAD_SIZE * lane_id, feat_end = min(THREAD_SIZE * (lane_id + 1), feat_in);

    if (feat_begin >= feat_end) return;

    for (int j = feat_begin; j < feat_end; ++j) {
        float result = 0.0f;
        for (int i = row_begin; i < row_end; ++i) {
            result +=  vin[idx[i] * feat_in + j] * val[i];
        }
        vout[warp_id * feat_in + j] = result;
    }
}

void SpMMOpt::preprocess(float *vin, float *vout)
{
    grid.x = (num_v + BLOCK_SIZE / WARP_SIZE - 1) / (BLOCK_SIZE / WARP_SIZE);
    block.x = BLOCK_SIZE;
}

void SpMMOpt::run(float *vin, float *vout)
{
    const int THREAD_SIZE = (feat_in + WARP_SIZE - 1) / WARP_SIZE;
    spmm_kernel<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in, THREAD_SIZE);
}