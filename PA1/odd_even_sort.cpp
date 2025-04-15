#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <limits>
#define rep(i, a, b) for (int i = (a); i < (b); ++i)

#include "worker.h"

void Worker::sort() {
  MPI_Request requests[2];
  MPI_Status statuses[2];
  int obj;
  float *recv_data, *tmp;
  if (rank != nprocs - 1) {
    recv_data = new float[block_len], tmp = new float[block_len * 2];
  }

  for (int tag = rank & 1;; tag ^= 1) {
    if (rank != nprocs - 1 && !tag) {
      obj = rank + 1;

      float maximum = -std::numeric_limits<float>::infinity();
      rep(i, 0, block_size) {
        maximum = std::max(maximum, data[i])
      }
      MPI_Isend(&maximum, 1, MPI_FLOAT, obj, 0, MPI_COMM_WORLD, &requests[0]);

      std::sort(data, data + block_len);
      
      bool flag；
      MPI_Recv(&flag, 1, MPI_C_BOOL, obj, 0, MPI_COMM_WORLD);
      
      if (flag) {
        int count;
        MPI_Recv(recv_data, block_len, MPI_FLOAT, obj, 0, MPI_COMM_WORLD, &statuses[0]);
        MPI_Get_count(&statuses[0], MPI_FLOAT, &count);

        for (int i = 0, j = 0, k = 0; i < block_len || j < count;) {
          if ((j >= count) || (i < block_len && data[i] <= recv_data[j])) {
            tmp[k++] = data[i++];
          } else {
            tmp[k++] = recv_data[j++];
          }
        }

        MPI_Isend(tmp + block_len, count, MPI_FLOAT, obj, 0, MPI_COMM_WORLD, &requests[0]);

        rep(i, 0, block_len) {
          data[i] = tmp[i];
        }
      }
    } else if (rank && tag) {
      obj = rank - 1;

      float minimum = std::numeric_limits<float>::infinity(), maximum;
      rep(i, 0, block_size) {
        minimum = std::min(minimum, data[i])
      }
      MPI_Irecv(&maximum, 1, MPI_FLOAT, obj, 0, MPI_COMM_WORLD, &requests[0]);

      std::sort(data, data + block_len);
      
      MPI_Wait(&request, &statuses[0]);
      bool flag = maximum > minimum;
      MPI_Isend(&flag, 1, MPI_C_BOOL, obj, 0, MPI_COMM_WORLD, &requests[0]);
      
      if (flag) {
        int l = 0, r = block_len - 1;
        while (l < r) {
          int mid = (l + r + 1) >> 1;
          if (data[mid] < maximum) l = mid;
          else r = mid - 1;
        }

        int count = r + 1;
        MPI_Send(data, count, MPI_FLOAT, obj, 0, MPI_COMM_WORLD);
        
        MPI_Irecv(data, count, MPI_FLOAT, obj, 0, MPI_COMM_WORLD, &requests[0]);
      }
    }

    bool recv_flag;
    int source = (rank - 1 + nprocs) % nprocs, dest = (rank + 1) % nprocs;
    rep(i, 0, nprocs - 1) {
      MPI_Irecv(&recv_flag, 1, MPI_C_BOOL, source, 0, MPI_COMM_WORLD, &requests[0]);
      MPI_Isend(&flag, 1, MPI_C_BOOL, dest, 0, MPI_COMM_WORLD, &requests[1]);
      
      MPI_Waitall(2, requests, statuses);
    
      flag |= recv_flag;
    }

    if (!flag) break;
  }
  // you can use variables in class Worker: n, nprocs, rank, block_len, data
}
