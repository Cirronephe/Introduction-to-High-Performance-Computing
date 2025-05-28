#include "spmm_opt.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <set>
#include <unordered_set>

const int BLOCK_SIZE = 128, WARP_SIZE = 32;

__global__ void spmm_kernel(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int feat_in, int *new_to_old)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id >= num_v) return;
    int row_id = new_to_old[warp_id];

    int row_begin = ptr[warp_id], row_end = ptr[warp_id + 1];

    for (int j = lane_id; j < feat_in; j += WARP_SIZE) {
        float result = 0.0f;
        for (int i = row_begin; i < row_end; ++i) {
            result +=  vin[idx[i] * feat_in + j] * val[i];
        }
        vout[row_id * feat_in + j] = result;
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

void SpMMOpt::read_metis_part(float *vin, const std::string& filename) {
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

    std::vector<int> degree(num_v);
    for (int i = 0; i < num_v; ++i) {
        degree[i] = h_ptr[i+1] - h_ptr[i];
    }

    std::sort(vertex_ids.begin(), vertex_ids.end(), [&](int a, int b) {
        if (part[a] != part[b]) 
            return part[a] < part[b];
        return degree[a] < degree[b];
    });

    old_to_new.resize(num_v);
    new_to_old.resize(num_v);
    for (int new_id = 0; new_id < num_v; ++new_id) {
        int old_id = vertex_ids[new_id];
        old_to_new[old_id] = new_id;
        new_to_old[new_id] = old_id;
    }
    cudaMalloc(&d_new_to_old, num_v * sizeof(int));
    cudaMemcpy(d_new_to_old, new_to_old.data(), num_v * sizeof(int), cudaMemcpyHostToDevice);

    std::vector<int> n_ptr;
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

        for (const auto &x : row) {
            n_idx.push_back(x.first);
            n_val.push_back(x.second);
            edge_count++;
        }
    }
    n_ptr.push_back(edge_count);

    cudaMemcpy(d_ptr, n_ptr.data(), (num_v + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_idx, n_idx.data(), num_e * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, n_val.data(), num_e * sizeof(float), cudaMemcpyHostToDevice);

    float *h_vin = new float[num_v * feat_in];
    float *n_vin = new float[num_v * feat_in];

    cudaMemcpy(h_vin, vin, num_v * feat_in * sizeof(float), cudaMemcpyDeviceToHost);

    for (int old_v = 0; old_v < num_v; ++old_v) {
        int new_v = old_to_new[old_v];
        for (int j = 0; j < feat_in; ++j) {
            n_vin[new_v * feat_in + j] = h_vin[old_v * feat_in + j];
        }
    }

    cudaMalloc(&p_vin, num_v * feat_in * sizeof(float));
    cudaMemcpy(p_vin, n_vin, num_v * feat_in * sizeof(float), cudaMemcpyHostToDevice);

    delete[] h_vin;
    delete[] n_vin;
}

void SpMMOpt::preprocess(float *vin, float *vout) {
    write_metis_graph("/home/course/hpc/users/2023010828/PA3/graph/citation.graph");
    return;
    // read_metis_part(vin, "/home/course/hpc/users/2023010828/PA3/part/collab.graph.part.16");
    grid.x = (num_v + BLOCK_SIZE / WARP_SIZE - 1) / (BLOCK_SIZE / WARP_SIZE);
    block.x = BLOCK_SIZE;
}

void SpMMOpt::run(float *vin, float *vout)
{
    // return;
    spmm_kernel<<<grid, block>>>(d_ptr, d_idx, d_val, p_vin, vout, num_v, feat_in, d_new_to_old);
}