#ifndef SpMM_OPT_H
#define SpMM_OPT_H
#include "spmm_base.h"

const int B = 32;

struct Tuple {
    int row;
    int count;
    int idx[B];
    float val[B];
};

class SpMMOpt : public SpMM
{
public:
    SpMMOpt(int *dev_out_ptr, int *dev_out_idx, int out_num_v, int out_num_e, int out_feat_in) : SpMM(dev_out_ptr, dev_out_idx, out_num_v, out_num_e, out_feat_in) {}
    SpMMOpt(CSR *g, int out_feat_in) : SpMM(g, out_feat_in) {}
    ~SpMMOpt() {
        if (target) checkCudaErrors(cudaFree(target));
        if (ptr_scheduled) checkCudaErrors(cudaFree(ptr_scheduled));
        if (d_new_to_old) checkCudaErrors(cudaFree(d_new_to_old));
        if (p_vin) checkCudaErrors(cudaFree(p_vin));
    }
     
    std::vector<Tuple> splitCSRToTuples();

    void uploadTuplesToGPU(const std::vector<Tuple> &hostTuples);

    virtual void preprocess(float *vin, float *vout);

    virtual void run(float *vin, float *vout);

    void write_metis_graph(const std::string& filename);

    void read_metis_part(float *vin, const std::string& filename);

    void reorder_vout(float *vout);

    void edgesort();

    void neighbor_grouping(int neighbor_num);

private:
    std::vector<int> old_to_new;
    std::vector<int> new_to_old;
    int *d_new_to_old;
    float *p_vin;
    Tuple *d_tuples;
    int tuple_count;

    int num_target;
    int *target, *ptr_scheduled;
};
#endif