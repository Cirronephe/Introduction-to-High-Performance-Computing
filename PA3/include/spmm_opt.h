#ifndef SpMM_OPT_H
#define SpMM_OPT_H
#include "spmm_base.h"

const int B = 32;

struct BLK {
    int row, cnt, idx[B];
    float val[B];
};

class SpMMOpt : public SpMM
{
public:
    SpMMOpt(int *dev_out_ptr, int *dev_out_idx, int out_num_v, int out_num_e, int out_feat_in) : SpMM(dev_out_ptr, dev_out_idx, out_num_v, out_num_e, out_feat_in) {}
    SpMMOpt(CSR *g, int out_feat_in) : SpMM(g, out_feat_in) {}
    ~SpMMOpt() {
        // if (d_new_to_old) checkCudaErrors(cudaFree(d_new_to_old));
        // if (p_vin) checkCudaErrors(cudaFree(p_vin));
        if (blk_row) checkCudaErrors(cudaFree(blk_row));
        if (blk_cnt) checkCudaErrors(cudaFree(blk_cnt));
        if (blk_idx) checkCudaErrors(cudaFree(blk_idx));
        if (blk_val) checkCudaErrors(cudaFree(blk_val));
    }
     
    std::vector<BLK> csr_to_blks();

    void cpu_to_gpu(const std::vector<BLK> &blks);

    virtual void preprocess(float *vin, float *vout);

    virtual void run(float *vin, float *vout);

    // void write_metis_graph(const std::string& filename);

    // void read_metis_part(float *vin, const std::string& filename);

private:
    // std::vector<int> old_to_new;
    // std::vector<int> new_to_old;
    // int *d_new_to_old;
    // float *p_vin;

    int blk_tot;
    int *blk_row, *blk_cnt, *blk_idx;
    float *blk_val;
};
#endif