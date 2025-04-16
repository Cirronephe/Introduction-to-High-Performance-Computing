#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mpi.h>
#define rep(i, a, b) for (int i = (a); i < (b); ++i)
#define EPS 1e-7
#define valid(a) ((a) >=0 && (a) < nprocs)

#include "worker.h"

void merge_split(float *data, float *recv_data, float *tmp, int len, int recv_len, int tag, int &bias) {
  std::merge(data, data + len, recv_data, recv_data + recv_len, tmp);

  if (!tag) {
    bias = 0;
  } else {
    bias = recv_len;
  }
} 

void Worker::sort() {
  MPI_Request requests[2];
  MPI_Status status, statuses[2];

  std::sort(data, data + block_len);

  int left = rank - 1, right = rank + 1, block_size = ceiling(n, nprocs), recv_count, bias = 0;
  bool flag;
  float maximum, minimum;
  float *recv_data = new float[block_size], *tmp[2];
  tmp[0] = new float[block_size + block_len];
  tmp[1] = new float[block_size + block_len];

  memcpy(tmp[0], data, block_len * sizeof(float));

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
          MPI_Sendrecv(tmp[i & 1] + bias, block_len, MPI_FLOAT, right, 0,
                       recv_data, block_size, MPI_FLOAT, right, 0,
                       MPI_COMM_WORLD, &status);
          MPI_Get_count(&status, MPI_FLOAT, &recv_count);

          minimum = std::min(tmp[i & 1][bias + 0], recv_data[0]);
        } else {
          minimum = tmp[i & 1][bias + 0];
        }
      } else {
        minimum = tmp[i & 1][bias + 0];
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
          MPI_Sendrecv(tmp[i & 1] + bias, block_len, MPI_FLOAT, left, 0,
                       recv_data, block_size, MPI_FLOAT, left, 0,
                       MPI_COMM_WORLD, &status);
          MPI_Get_count(&status, MPI_FLOAT, &recv_count);

          maximum = std::max(tmp[i & 1][bias + block_len - 1], recv_data[recv_count - 1]);
        } else {
          maximum = tmp[i & 1][bias + block_len - 1];
        }
      } else {
        maximum = tmp[i & 1][bias + block_len - 1];
      }

      if (i != nprocs - 1 && valid(right)) {
        MPI_Irecv(&minimum, 1, MPI_FLOAT, right, 0, MPI_COMM_WORLD, &requests[0]);
        MPI_Isend(&maximum, 1, MPI_FLOAT, right, 0, MPI_COMM_WORLD, &requests[1]);
      }
    }

    if (flag) {
      merge_split(tmp[i & 1] + bias, recv_data, tmp[(i & 1) ^ 1], block_len, recv_count, tag, bias);
    }
  }

  memcpy(data, tmp[nprocs & 1] + bias, block_len * sizeof(float));

  delete[] recv_data;
  delete[] tmp[0];
  delete[] tmp[1];
}
