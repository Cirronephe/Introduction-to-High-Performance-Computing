#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#define rep(i, a, b) for (int i = (a); i < (b); ++i)
#define EPS 1e-7

#include "worker.h"

void merge_split(float *data, float *recv_data, float *tmp, int len, int recv_len, bool flag) {
  std::merge(data, data + len, recv_data, recv_data + recv_len, tmp);

  if (!flag) {
    memcpy(data, tmp, len * sizeof(float));
  } else {
    memcpy(data, tmp + recv_len, len * sizeof(float));
  }
}

void Worker::sort() {
  MPI_Request request;
  MPI_Status status;

  std::sort(data, data + block_len);

  int obj, block_size = ceiling(n, nprocs);
  bool flag;
  float *recv_data = new float[block_size], *tmp = new float[block_size + block_len];

  for (int i = 0, tag = rank & 1; i < nprocs; ++i, tag ^= 1) {
    if (rank != nprocs - 1 && !tag) {
      obj = rank + 1;
      float maximum = data[block_len - 1], minimum;
      MPI_Sendrecv(&maximum, 1, MPI_FLOAT, obj, 0, 
                   &minimum, 1, MPI_FLOAT, obj, 0,
                   MPI_COMM_WORLD, &status);
      flag = (maximum - minimum) > EPS;
      
      if (flag) {
        MPI_Sendrecv(data, block_len, MPI_FLOAT, obj, 0,
                     recv_data, block_size, MPI_FLOAT, obj, 0,
                     MPI_COMM_WORLD, &status);
        int recv_count;
        MPI_Get_count(&status, MPI_FLOAT, &recv_count);

        merge_split(data, recv_data, tmp, block_len, recv_count, false);
      }
    } else if (rank && tag) {
      obj = rank - 1;
      float minimum = data[0], maximum;
      MPI_Sendrecv(&minimum, 1, MPI_FLOAT, obj, 0, 
                   &maximum, 1, MPI_FLOAT, obj, 0,
                   MPI_COMM_WORLD, &status);
      flag = (maximum - minimum) > EPS;
      
      if (flag) {
        MPI_Sendrecv(data, block_len, MPI_FLOAT, obj, 0,
                     recv_data, block_size, MPI_FLOAT, obj, 0,
                     MPI_COMM_WORLD, &status);
        int recv_count;
        MPI_Get_count(&status, MPI_FLOAT, &recv_count);

        merge_split(data, recv_data, tmp, block_len, recv_count, true);
      }
    }

    if (request != NULL) MPI_Wait(&request, &status);
  }
  // you can use variables in class Worker: n, nprocs, rank, block_len, data
}
