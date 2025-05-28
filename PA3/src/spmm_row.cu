#include "spmm_opt.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

const int BLOCK_SIZE = 128, WARP_SIZE = 32;

__global__ void spmm_kernel(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int feat_in)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id >= num_v) return;

    int row_begin = ptr[warp_id], row_end = ptr[warp_id + 1];

    for (int j = lane_id; j < feat_in; j += WARP_SIZE) {
        float result = 0.0f;
        for (int i = row_begin; i < row_end; ++i) {
            result +=  vin[idx[i] * feat_in + j] * val[i];
        }
        vout[warp_id * feat_in + j] = result;
    }
}

void SpMMOpt::write_metis_graph(const std::string& filename) {
    std::vector<int> h_ptr(num_v + 1);
    std::vector<int> h_idx(num_e);

    cudaMemcpy(h_ptr.data(), d_ptr, (num_v + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_idx.data(), d_idx, num_e * sizeof(int), cudaMemcpyDeviceToHost);
    
    std::vector<std::vector<int>> adj(num_v);

    for (int u = 0; u < num_v; ++u) {
        for (int eid = h_ptr[u]; eid < h_ptr[u + 1]; ++eid) {
            int v = h_idx[eid];
            adj[u].push_back(v);
        }
    }

    std::ofstream ofs(filename);
    ofs << num_v << " " << num_e / 2 << "\n";
    for (int u = 0; u < num_v; ++u) {
        bool first = true;
        for (int v : adj[u]) {
            if (!first) ofs << " ";
            ofs << (v + 1);
            first = false;
        }
        ofs << "\n";
    }
    ofs.close();
}

void SpMMOpt::read_metis_part(const std::string& filename) {
    std::vector<int> h_ptr(num_v + 1);
    std::vector<int> h_idx(num_e);
    std::vector<float> h_val(num_e);

    cudaMemcpy(h_ptr.data(), d_ptr, (num_v + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_idx.data(), d_idx, num_e * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_val.data(), d_val, num_e * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<int> part(num_v);
    std::ifstream ifs(filename);
    for (int i = 0; i < num_v; ++i) {
        ifs >> part[i];
    }
    ifs.close();

    std::vector<int> vertex_ids(num_v);
    for (int i = 0; i < num_v; ++i) vertex_ids[i] = i;

    std::sort(vertex_ids.begin(), vertex_ids.end(), [&](int a, int b) {
        return part[a] < part[b];
    });

    old_to_new.resize(num_v);
    new_to_old.resize(num_v);
    for (int new_id = 0; new_id < num_v; ++new_id) {
        int old_id = vertex_ids[new_id];
        old_to_new[old_id] = new_id;
        new_to_old[new_id] = old_id;
    }

    std::vector<int> n_idx;
    std::vector<float> n_val;

    int edge_count = 0;
    for (int new_u = 0; new_u < num_v; ++new_u) {
        int old_u = new_to_old[new_u];
        n_ptr.push_back(edge_count);

        std::vector<std::pair<int, float>> row;
        for (int eid = h_ptr[old_u]; eid < h_ptr[old_u + 1]; ++eid) {
            int old_v = h_idx[eid];
            int new_v = old_to_new[old_v];
            row.emplace_back(new_v, h_val[eid]);
        }

        std::sort(row.begin(), row.end());

        for (const auto &[v, val] : row) {
            n_idx.push_back(v);
            n_val.push_back(val);
            edge_count++;
        }
    }
    n_ptr.push_back(edge_count); // num_e == edge_count

    cudaMemcpy(d_ptr, n_ptr.data(), (num_v + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_idx, n_idx.data(), num_e * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, n_val.data(), num_e * sizeof(float), cudaMemcpyHostToDevice);
}

void SpMMOpt::preprocess(float *vin, float *vout) {
    // write_metis_graph("/home/course/hpc/users/2023010828/PA3/graph/graph.graph");
    read_metis_part("/home/course/hpc/users/2023010828/PA3/part/graph.graph.part.4");
    grid.x = (num_v + BLOCK_SIZE / WARP_SIZE - 1) / (BLOCK_SIZE / WARP_SIZE);
    block.x = BLOCK_SIZE;
}

void SpMMOpt::run(float *vin, float *vout)
{
    spmm_kernel<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
}