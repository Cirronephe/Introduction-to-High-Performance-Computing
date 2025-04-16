#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#define rep(i, a, b) for (int i = (a); i < (b); ++i)
#define EPS 1e-7

#include "worker.h"

void Worker::sort() {
  MPI_Request request;
  MPI_Status status;

  std::sort(data, data + block_len);

  int obj;
  bool flag;
  float *recv_data = new float[block_len], *tmp = new float[block_len * 2];
  int cnt = 0;
  for (int tag = rank & 1;; tag ^= 1) {
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
          if ((j >= count) || (i < int(block_len) && (data[i] - recv_data[j]) < EPS)) {
            tmp[k++] = data[i++];
          } else {
            tmp[k++] = recv_data[j++];
          }
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
        }

        int count = r + 1;
        MPI_Send(data, count, MPI_FLOAT, obj, 0, MPI_COMM_WORLD);
        
        MPI_Irecv(data, count, MPI_FLOAT, obj, 0, MPI_COMM_WORLD, &request);
      }
    }
 
    // int source = (rank - 1 + nprocs) % nprocs, dest = (rank + 1) % nprocs;

    // rep(i, 0, nprocs - 1) {
    //   MPI_Irecv(&recv_flag, 1, MPI_C_BOOL, source, 0, MPI_COMM_WORLD, &requests[0]);
    //   MPI_Isend(&flag, 1, MPI_C_BOOL, dest, 0, MPI_COMM_WORLD, &requests[1]);
      
    //   MPI_Waitall(2, requests, statuses);
    
    //   flag |= recv_flag;
    // }

    if (request != NULL) MPI_Wait(&request, &status);

    // if (!flag) break;

    if (++cnt == nprocs) break;
  }
  // you can use variables in class Worker: n, nprocs, rank, block_len, data
}
