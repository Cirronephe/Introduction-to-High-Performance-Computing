#include "spmm_opt.h"

__global__ void spmm_kernel(int *blk_row, int *blk_cnt, int *blk_idx, float *blk_val, 
                            float *vin, float *vout, int feat_in) {
    int bid = blockIdx.x, tid = threadIdx.x;

    __shared__ int s_idx[B];
    __shared__ float s_val[B];
    int row = blk_row[bid], cnt = blk_cnt[bid];

    if (tid < cnt) {
        s_idx[tid] = blk_idx[bid * B + tid];
        s_val[tid] = blk_val[bid * B + tid];
    }

    __syncthreads();

    float result = 0.0f;
    for (int i = 0; i < cnt; ++i) {
        result += vin[s_idx[i] * feat_in + tid] * s_val[i];
    }
    atomicAdd(&vout[row * feat_in + tid], result);
}

std::vector<BLK> SpMMOpt::csr_to_blks() {
    std::vector<int> h_ptr(num_v + 1);
    std::vector<int> h_idx(num_e);
    std::vector<float> h_val(num_e);

    cudaMemcpy(h_ptr.data(), d_ptr, (num_v + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_idx.data(), d_idx, num_e * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_val.data(), d_val, num_e * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<BLK> blks;

    for (int row = 0; row < num_v; ++row) {
        int row_start = h_ptr[row], row_end = h_ptr[row + 1];
        int nnz = row_end - row_start;

        int blk_num = (nnz + B - 1) / B;

        for (int i = 0; i < blk_num; ++i) {
            BLK blk;
            blk.row = row;
            blk.cnt = std::min(B, nnz - i * B);

            for (int j = 0; j < blk.cnt; ++j) {
                int eid = row_start + i * B + j;
                blk.idx[j] = h_idx[eid];
                blk.val[j] = h_val[eid];
            }

            blks.push_back(blk);
        }
    }

    return blks;
}

void SpMMOpt::cpu_to_gpu(const std::vector<BLK> &blks) {
    std::vector<int> h_blk_row(blk_tot);
    std::vector<int> h_blk_cnt(blk_tot);
    std::vector<int> h_blk_idx(blk_tot * B);
    std::vector<float> h_blk_val(blk_tot * B);

    for (int i = 0; i < blk_tot; ++i) {
        h_blk_row[i] = blks[i].row;
        h_blk_cnt[i] = blks[i].cnt;
        for (int j = 0; j < B; ++j) {
            h_blk_idx[i * B + j] = blks[i].idx[j];
            h_blk_val[i * B + j] = blks[i].val[j];
        }
    }

    cudaMalloc(&blk_row, blk_tot * sizeof(int));
    cudaMemcpy(blk_row, h_blk_row.data(), blk_tot * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&blk_cnt, blk_tot * sizeof(int));
    cudaMemcpy(blk_cnt, h_blk_cnt.data(), blk_tot * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&blk_idx, blk_tot * B * sizeof(int));
    cudaMemcpy(blk_idx, h_blk_idx.data(), blk_tot * B * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&blk_val, blk_tot * B * sizeof(float));
    cudaMemcpy(blk_val, h_blk_val.data(), blk_tot * B * sizeof(float), cudaMemcpyHostToDevice);
}

void SpMMOpt::preprocess(float *vin, float *vout)
{
    std::vector<BLK> blks = csr_to_blks();
    blk_tot = blks.size();
    cpu_to_gpu(blks);

    grid.x = blk_tot;
    block.x = feat_in;
}

void SpMMOpt::run(float *vin, float *vout)
{
    spmm_kernel<<<grid, block>>>(blk_row, blk_cnt, blk_idx, blk_val, vin, vout, feat_in);
}


// void SpMMOpt::write_metis_graph(const std::string& filename) {
//     std::vector<int> h_ptr(num_v + 1);
//     std::vector<int> h_idx(num_e);

//     cudaMemcpy(h_ptr.data(), d_ptr, (num_v + 1) * sizeof(int), cudaMemcpyDeviceToHost);
//     cudaMemcpy(h_idx.data(), d_idx, num_e * sizeof(int), cudaMemcpyDeviceToHost);
    
//     std::vector<std::vector<int>> adj(num_v);

//     for (int u = 0; u < num_v; ++u) {
//         for (int eid = h_ptr[u]; eid < h_ptr[u + 1]; ++eid) {
//             int v = h_idx[eid];
//             adj[u].push_back(v);
//         }
//     }

//     std::ofstream ofs(filename);
//     ofs << num_v << " " << num_e / 2 << "\n";
//     for (int u = 0; u < num_v; ++u) {
//         bool first = true;
//         for (int v : adj[u]) {
//             if (!first) ofs << " ";
//             ofs << (v + 1);
//             first = false;
//         }
//         ofs << "\n";
//     }
//     ofs.close();
// }

// void SpMMOpt::read_metis_part(float *vin, const std::string& filename) {
//     std::vector<int> h_ptr(num_v + 1);
//     std::vector<int> h_idx(num_e);
//     std::vector<float> h_val(num_e);

//     cudaMemcpy(h_ptr.data(), d_ptr, (num_v + 1) * sizeof(int), cudaMemcpyDeviceToHost);
//     cudaMemcpy(h_idx.data(), d_idx, num_e * sizeof(int), cudaMemcpyDeviceToHost);
//     cudaMemcpy(h_val.data(), d_val, num_e * sizeof(float), cudaMemcpyDeviceToHost);

//     std::vector<int> part(num_v);
//     std::ifstream ifs(filename);
//     for (int i = 0; i < num_v; ++i) {
//         ifs >> part[i];
//     }
//     ifs.close();

//     std::vector<int> vertex_ids(num_v);
//     for (int i = 0; i < num_v; ++i) vertex_ids[i] = i;

//     std::vector<int> degree(num_v);
//     for (int i = 0; i < num_v; ++i) {
//         degree[i] = h_ptr[i+1] - h_ptr[i];
//     }

//     std::sort(vertex_ids.begin(), vertex_ids.end(), [&](int a, int b) {
//         if (part[a] != part[b]) 
//             return part[a] < part[b];
//         return degree[a] < degree[b];
//     });

//     old_to_new.resize(num_v);
//     new_to_old.resize(num_v);
//     for (int new_id = 0; new_id < num_v; ++new_id) {
//         int old_id = vertex_ids[new_id];
//         old_to_new[old_id] = new_id;
//         new_to_old[new_id] = old_id;
//     }
//     cudaMalloc(&d_new_to_old, num_v * sizeof(int));
//     cudaMemcpy(d_new_to_old, new_to_old.data(), num_v * sizeof(int), cudaMemcpyHostToDevice);

//     std::vector<int> n_ptr;
//     std::vector<int> n_idx;
//     std::vector<float> n_val;

//     int edge_count = 0;
//     for (int new_u = 0; new_u < num_v; ++new_u) {
//         int old_u = new_to_old[new_u];
//         n_ptr.push_back(edge_count);

//         std::vector<std::pair<int, float>> row;
//         for (int eid = h_ptr[old_u]; eid < h_ptr[old_u + 1]; ++eid) {
//             int old_v = h_idx[eid];
//             int new_v = old_to_new[old_v];
//             row.emplace_back(new_v, h_val[eid]);
//         }

//         std::sort(row.begin(), row.end());

//         for (const auto &x : row) {
//             n_idx.push_back(x.first);
//             n_val.push_back(x.second);
//             edge_count++;
//         }
//     }
//     n_ptr.push_back(edge_count);

//     cudaMemcpy(d_ptr, n_ptr.data(), (num_v + 1) * sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_idx, n_idx.data(), num_e * sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_val, n_val.data(), num_e * sizeof(float), cudaMemcpyHostToDevice);

//     float *h_vin = new float[num_v * feat_in];
//     float *n_vin = new float[num_v * feat_in];

//     cudaMemcpy(h_vin, vin, num_v * feat_in * sizeof(float), cudaMemcpyDeviceToHost);

//     for (int old_v = 0; old_v < num_v; ++old_v) {
//         int new_v = old_to_new[old_v];
//         for (int j = 0; j < feat_in; ++j) {
//             n_vin[new_v * feat_in + j] = h_vin[old_v * feat_in + j];
//         }
//     }

//     cudaMalloc(&p_vin, num_v * feat_in * sizeof(float));
//     cudaMemcpy(p_vin, n_vin, num_v * feat_in * sizeof(float), cudaMemcpyHostToDevice);

//     delete[] h_vin;
//     delete[] n_vin;
// }