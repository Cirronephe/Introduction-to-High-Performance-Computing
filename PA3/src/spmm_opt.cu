#include "spmm_opt.h"

__global__ void spmm_tuple_kernel(Tuple *tuples, float *vin, float *vout, int feat_in) {
    int bid = blockIdx.x, tid = threadIdx.x;

    __shared__ int s_idx[B];
    __shared__ float s_val[B];

    int cnt = tuples[tuple_id].count;

    if (feat_id < cnt) {
        s_idx[feat_id] = tuples[tuple_id].idx[feat_id];
        s_val[feat_id] = tuples[tuple_id].val[feat_id];
    }

    __syncthreads();

    if (feat_id >= feat_in) return;

    float result = 0.0f;

    for (int i = 0; i < cnt; ++i) {
        int col = s_idx[i];
        float weight = s_val[i];
        result += vin[col * feat_in + feat_id] * weight;
    }

    atomicAdd(&vout[tuples[tuple_id].row * feat_in + feat_id], result);
}

std::vector<Tuple> SpMMOpt::splitCSRToTuples() {
    std::vector<int> h_ptr(num_v + 1);
    std::vector<int> h_idx(num_e);
    std::vector<float> h_val(num_e);

    cudaMemcpy(h_ptr.data(), d_ptr, (num_v + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_idx.data(), d_idx, num_e * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_val.data(), d_val, num_e * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<Tuple> tuples;

    for (int row = 0; row < num_v; ++row) {
        int row_start = h_ptr[row], row_end = h_ptr[row + 1];
        int nnz = row_end - row_start;

        int tuple_num = (nnz + B - 1) / B;

        for (int t = 0; t < tuple_num; ++t) {
            Tuple tuple;
            tuple.row = row;
            tuple.count = std::min(B, nnz - t * B);

            for (int j = 0; j < tuple.count; ++j) {
                int global_idx = row_start + t * B + j;
                tuple.idx[j] = h_idx[global_idx];
                tuple.val[j] = h_val[global_idx];
            }

            for (int j = tuple.count; j < B; ++j) {
                tuple.idx[j] = -1;
                tuple.val[j] = 0.0f;
            }

            tuples.push_back(tuple);
        }
    }

    return tuples;
}

void SpMMOpt::uploadTuplesToGPU(const std::vector<Tuple> &hostTuples) {
    cudaMalloc(&d_tuples, hostTuples.size() * sizeof(Tuple));
    cudaMemcpy(d_tuples, hostTuples.data(), hostTuples.size() * sizeof(Tuple), cudaMemcpyHostToDevice);
}

void SpMMOpt::preprocess(float *vin, float *vout)
{
    std::vector<Tuple> hostTuples = splitCSRToTuples();
    uploadTuplesToGPU(hostTuples);
    std::sort(hostTuples.begin(), hostTuples.end(), [](const Tuple& a, const Tuple& b) {
        return a.idx[0] < b.idx[0];
    });
    tuple_count = hostTuples.size();
    grid.x = tuple_count; // (num_v + BLOCK_SIZE / WARP_SIZE - 1) / (BLOCK_SIZE / WARP_SIZE);
    block.x = feat_in; // ->>>>
}

void SpMMOpt::run(float *vin, float *vout)
{
    // spmm_kernel<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
    spmm_tuple_kernel<<<grid, block>>>(d_tuples, vin, vout, feat_in);
}