#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#define rep(i, a, b) for (int i = (a); i < (b); ++i)
#define EPS 1e-7
#define valid(a) ((a) >=0 && (a) < nprocs)

#include "worker.h"

void merge_split(float *data, float *recv_data, float *tmp, int len, int recv_len, int tag) {
  std::merge(data, data + len, recv_data, recv_data + recv_len, tmp);

  if (!tag) {
    memcpy(data, tmp, len * sizeof(float));
  } else {
    memcpy(data, tmp + recv_len, len * sizeof(float));
  }
}

void Worker::sort() {
  MPI_Request requests[2];
  MPI_Status status, statuses[2];

  std::sort(data, data + block_len);

  int left = rank - 1, right = rank + 1, block_size = ceiling(n, nprocs), recv_count;
  bool flag;
  float maximum, minimum;
  float *recv_data = new float[block_size], *tmp = new float[block_size + block_len];

  for (int i = 0, tag = rank & 1; i < nprocs; ++i, tag ^= 1) {
    flag = false;

    if (!tag) {
      if (valid(right)) {
        if (!i) {
          maximum = data[block_len - 1];
          MPI_Irecv(&minimum, 1, MPI_FLOAT, right, 0, MPI_COMM_WORLD, &requests[0]);
          MPI_Isend(&maximum, 1, MPI_FLOAT, right, 0, MPI_COMM_WORLD, &requests[1]);
        }
        
        MPI_Waitall(2, requests, statuses);
        flag = (maximum - minimum) > EPS;
        
        if (flag) {
          MPI_Sendrecv(data, block_len, MPI_FLOAT, right, 0,
                      recv_data, block_size, MPI_FLOAT, right, 0,
                      MPI_COMM_WORLD, &status);
          MPI_Get_count(&status, MPI_FLOAT, &recv_count);

          minimum = std::min(data[0], recv_data[0]);
        } else {
          minimum = data[0];
        }
      } else {
        minimum = data[0];
      }
      
      if (i != nprocs - 1 && valid(left)) {
        MPI_Irecv(&maximum, 1, MPI_FLOAT, left, 0, MPI_COMM_WORLD, &requests[0]);
        MPI_Isend(&minimum, 1, MPI_FLOAT, left, 0, MPI_COMM_WORLD, &requests[1]);
      }
    } else {
      if (valid(left)) {
        if (!i) {
          minimum = data[0];
          MPI_Irecv(&maximum, 1, MPI_FLOAT, left, 0, MPI_COMM_WORLD, &requests[0]);
          MPI_Isend(&minimum, 1, MPI_FLOAT, left, 0, MPI_COMM_WORLD, &requests[1]);
        }
        
        MPI_Waitall(2, requests, statuses);
        flag = (maximum - minimum) > EPS;
        
        if (flag) {
          MPI_Sendrecv(data, block_len, MPI_FLOAT, left, 0,
                      recv_data, block_size, MPI_FLOAT, left, 0,
                      MPI_COMM_WORLD, &status);
          MPI_Get_count(&status, MPI_FLOAT, &recv_count);

          maximum = std::max(data[block_len - 1], recv_data[recv_count - 1]);
        } else {
          maximum = data[block_len - 1];
        }
      } else {
        maximum = data[block_len - 1];
      }

      if (i != nprocs - 1 && valid(right)) {
        MPI_Irecv(&minimum, 1, MPI_FLOAT, right, 0, MPI_COMM_WORLD, &requests[0]);
        MPI_Isend(&maximum, 1, MPI_FLOAT, right, 0, MPI_COMM_WORLD, &requests[1]);
      }
    }

    if (flag) {
      merge_split(data, recv_data, tmp, block_len, recv_count, tag);
    }
  }
  // you can use variables in class Worker: n, nprocs, rank, block_len, data
}
