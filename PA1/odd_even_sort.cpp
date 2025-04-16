#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#define rep(i, a, b) for (int i = (a); i < (b); ++i)
#define EPS 1e-7

#include "worker.h"

void pass(int nprocs, int source, int dest, MPI_Request *requests, MPI_Status *statuses, 
          int *cnts, bool &flag, bool &recv_flag) {
  if (cnts[1] == nprocs - 1) return;

  if (!cnts[0]) {
    ++cnts[0];
    MPI_Irecv(&recv_flag, 1, MPI_C_BOOL, source, 1, MPI_COMM_WORLD, &requests[0]);
    MPI_Isend(&flag, 1, MPI_C_BOOL, dest, 1, MPI_COMM_WORLD, &requests[1]);
  } else {
    int test_flag;
    MPI_Testall(2, requests, &test_flag, statuses);

    if (test_flag) {
      flag |= recv_flag;
      ++cnts[1];
      
      if (cnts[0] < nprocs - 1) {
        ++cnts[0];
        MPI_Irecv(&recv_flag, 1, MPI_C_BOOL, source, 1, MPI_COMM_WORLD, &requests[0]);
        MPI_Isend(&flag, 1, MPI_C_BOOL, dest, 1, MPI_COMM_WORLD, &requests[1]);
      }
    }
  }
}

void Worker::sort() {
  MPI_Request request, requests[2];
  MPI_Status status, statuses[2];

  std::sort(data, data + block_len);

  int obj, cnts[2], source = (rank - 1 + nprocs) % nprocs, dest = (rank + 1) % nprocs;
  bool flag, recv_flag;
  float *recv_data = new float[block_len], *tmp = new float[block_len * 2];

  for (int tag = rank & 1;; tag ^= 1) {
    cnts[0] = cnts[1] = 0;

    if (rank != nprocs - 1 && !tag) {
      obj = rank + 1;
      float maximum = data[block_len - 1];
      MPI_Send(&maximum, 1, MPI_FLOAT, obj, 0, MPI_COMM_WORLD);
      MPI_Recv(&flag, 1, MPI_C_BOOL, obj, 0, MPI_COMM_WORLD, &status);

      if (flag) {
        int count;
        MPI_Recv(recv_data, block_len, MPI_FLOAT, obj, 0, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_FLOAT, &count);

        for (int i = 0, j = 0, k = 0; i < int(block_len) || j < count;) {
          if ((j == count) || (i < int(block_len) && (data[i] - recv_data[j]) < EPS)) {
            tmp[k++] = data[i++];
          } else {
            tmp[k++] = recv_data[j++];
          }
          pass(nprocs, source, dest, requests, statuses, cnts, flag, recv_flag);
        }

        MPI_Isend(tmp + block_len, count, MPI_FLOAT, obj, 0, MPI_COMM_WORLD, &request);

        memcpy(data, tmp, block_len * sizeof(float));
      }
    } else if (rank && tag) {
      obj = rank - 1;
      float minimum = data[0], maximum;
      MPI_Recv(&maximum, 1, MPI_FLOAT, obj, 0, MPI_COMM_WORLD, &status);
      flag = (maximum - minimum) > EPS;
      MPI_Isend(&flag, 1, MPI_C_BOOL, obj, 0, MPI_COMM_WORLD, &request);
      
      if (flag) {
        int l = 0, r = block_len - 1;
        while (l < r) {
          int mid = (l + r + 1) >> 1;
          if ((data[mid] - maximum) < -EPS) l = mid;
          else r = mid - 1;
          pass(nprocs, source, dest, requests, statuses, cnts, flag, recv_flag);
        }

        int count = r + 1;
        MPI_Send(data, count, MPI_FLOAT, obj, 0, MPI_COMM_WORLD);
        
        MPI_Irecv(data, count, MPI_FLOAT, obj, 0, MPI_COMM_WORLD, &request);
      }
    }

    rep(i, cnts[1], nprocs - 1) {
      if (cnts[0] == cnts[1]) {
        ++cnts[0];
        MPI_Irecv(&recv_flag, 1, MPI_C_BOOL, source, 1, MPI_COMM_WORLD, &requests[0]);
        MPI_Isend(&flag, 1, MPI_C_BOOL, dest, 1, MPI_COMM_WORLD, &requests[1]);
      }
      
      MPI_Waitall(2, requests, statuses);

      flag |= recv_flag;
      ++cnts[1];
    }

    if (request != NULL) MPI_Wait(&request, &status);

    if (!flag) break;
  }
  // you can use variables in class Worker: n, nprocs, rank, block_len, data
}
