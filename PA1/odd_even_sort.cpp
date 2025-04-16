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

void half_merge(float *data, float *recv_data, float *tmp, int len, int recv_len, int tag) {
  if (!tag) {
    for (int i = 0, j = 0, k = 0; k < len;) {
      if ((j == recv_len) || (i < len && (data[i] - recv_data[j]) < EPS)) {
        tmp[k++] = data[i++];
      } else {
        tmp[k++] = recv_data[j++];
      }
    }
  } else {
    for (int i = len - 1, j = recv_len - 1, k = len - 1; k >= 0;) {
      if ((j < 0) || (i >= 0 && (data[i] - recv_data[j]) > -EPS)) {
        tmp[k--] = data[i--];
      } else {
        tmp[k--] = recv_data[j--];
      }
    }
  }
}

void Worker::sort() {
  MPI_Request requests[2];
  MPI_Status status, statuses[2];

  std::sort(data, data + block_len);

  int j = 0, left = rank - 1, right = rank + 1, block_size = ceiling(n, nprocs), recv_count;
  bool flag;
  float maximum, minimum;
  float *recv_data = new float[block_size], *tmp[2];
  tmp[0] = new float[block_len];
  tmp[1] = new float[block_len];

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
          int l = 0, r = block_len - 1;
          while (l < r) {
            int mid = (l + r) >> 1;
            if ((tmp[j][mid] - minimum) > EPS) r = mid;
            else l = mid + 1;
          }

          MPI_Sendrecv(tmp[j] + l, block_len - l, MPI_FLOAT, right, 0,
                       recv_data, block_size, MPI_FLOAT, right, 0,
                       MPI_COMM_WORLD, &status);
          MPI_Get_count(&status, MPI_FLOAT, &recv_count);

          minimum = std::min(tmp[j][0], recv_data[0]);
        } else {
          minimum = tmp[j][0];
        }
      } else {
        minimum = tmp[j][0];
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
          int l = 0, r = block_len - 1;
          while (l < r) {
            int mid = (l + r + 1) >> 1;
            if ((tmp[j][mid] - maximum) < -EPS) l = mid;
            else r = mid - 1;
          }

          MPI_Sendrecv(tmp[j], r + 1, MPI_FLOAT, left, 0,
                       recv_data, block_size, MPI_FLOAT, left, 0,
                       MPI_COMM_WORLD, &status);
          MPI_Get_count(&status, MPI_FLOAT, &recv_count);

          maximum = std::max(tmp[j][block_len - 1], recv_data[recv_count - 1]);
        } else {
          maximum = tmp[j][block_len - 1];
        }
      } else {
        maximum = tmp[j][block_len - 1];
      }

      if (i != nprocs - 1 && valid(right)) {
        MPI_Irecv(&minimum, 1, MPI_FLOAT, right, 0, MPI_COMM_WORLD, &requests[0]);
        MPI_Isend(&maximum, 1, MPI_FLOAT, right, 0, MPI_COMM_WORLD, &requests[1]);
      }
    }
 
    if (flag) {
      half_merge(tmp[j], recv_data, tmp[j ^ 1], block_len, recv_count, tag);
      j ^= 1;
    }
  }

  memcpy(data, tmp[j], block_len * sizeof(float));

  delete[] recv_data;
  delete[] tmp[0];
  delete[] tmp[1];
}
