# 奇偶排序 大作业 实验报告 

颜子俊 2023010828

## 实现代码

```cpp
#define rep(i, a, b) for (int i = (a); i < (b); ++i)
#define EPS 1e-7
#define valid(a) ((a) >=0 && (a) < valid_procs) // 判断进程是否有效

/*
 * 对当前进程和交互进程的数据进行归并（仅保留必要部分）
 */
void half_merge(float *data, float *recv_data, float *tmp, int len, int recv_len, int tag) {
  if (!tag) { // 若当前进程位置靠左
    for (int i = 0, j = 0, k = 0; k < len;) { // 归并排序
      if ((j == recv_len) || (i < len && (data[i] - recv_data[j]) < EPS)) {
        tmp[k++] = data[i++];
      } else {
        tmp[k++] = recv_data[j++];
      }
    }
  } else { // 若当前进程位置靠右
    for (int i = len - 1, j = recv_len - 1, k = len - 1; k >= 0;) { // 从右侧开始的归并排序
      if ((j < 0) || (i >= 0 && (data[i] - recv_data[j]) > -EPS)) {
        tmp[k--] = data[i--];
      } else {
        tmp[k--] = recv_data[j--];
      }
    }
  }
}

void Worker::sort() {
  if (out_of_range) return;

  MPI_Request requests[2];
  MPI_Status status, statuses[2];

  std::sort(data, data + block_len); // 进程内排序

  int j = 0, left = rank - 1, right = rank + 1, recv_count,
      block_size = ceiling(n, nprocs), valid_procs = ceiling(n, block_size);
  bool flag; // 与交互进程是否需要进行排序
  float maximum, minimum; // 当前进程的最大值与最小值
  float *recv_data = new float[block_size], *tmp[2]; // 接受域，数据暂存域
  tmp[0] = new float[block_len]; // 省去反复拷贝数据的过程
  tmp[1] = new float[block_len];

  memcpy(tmp[0], data, block_len * sizeof(float));

  for (int i = 0, tag = rank & 1; i < valid_procs; ++i, tag ^= 1) { // 进行 valid_procs 次迭代后必然完成排序
    flag = false;

    if (!tag) { // 该进程位于左侧
      if (valid(right)) {
        if (!i) { // 判断是否需要进行排序
          maximum = data[block_len - 1];
          MPI_Irecv(&minimum, 1, MPI_FLOAT, right, 0, MPI_COMM_WORLD, &requests[0]);
          MPI_Isend(&maximum, 1, MPI_FLOAT, right, 0, MPI_COMM_WORLD, &requests[1]);
        }
        
        MPI_Waitall(2, requests, statuses);
        flag = (maximum - minimum) > EPS;
        
        if (flag) { // 需要排序
          int l = 0, r = block_len - 1; // 二分查找必要传输位置
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
      
      if (i != valid_procs - 1 && valid(left)) { // 使下一轮判断通信与后续合并计算重叠
        MPI_Irecv(&maximum, 1, MPI_FLOAT, left, 0, MPI_COMM_WORLD, &requests[0]);
        MPI_Isend(&minimum, 1, MPI_FLOAT, left, 0, MPI_COMM_WORLD, &requests[1]);
      }
    } else { // 该进程位于右侧
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

      if (i != valid_procs - 1 && valid(right)) {
        MPI_Irecv(&minimum, 1, MPI_FLOAT, right, 0, MPI_COMM_WORLD, &requests[0]);
        MPI_Isend(&maximum, 1, MPI_FLOAT, right, 0, MPI_COMM_WORLD, &requests[1]);
      }
    }
 
    if (flag) { // 如果需要进行排序，则归并两进程数据
      half_merge(tmp[j], recv_data, tmp[j ^ 1], block_len, recv_count, tag);
      j ^= 1; // 规避大量拷贝操作
    }
  }

  memcpy(data, tmp[j], block_len * sizeof(float)); // 将数据返还至原位

  delete[] recv_data;
  delete[] tmp[0];
  delete[] tmp[1];
}
```

## 性能优化方式

### 空间换时间规避大量拷贝

```cpp
float *recv_data = new float[block_size], *tmp[2]; // 接受域，数据暂存域
tmp[0] = new float[block_len]; // 省去反复拷贝数据的过程
tmp[1] = new float[block_len];
```

对于 56 进程 $n = 10^8$，从 1100ms 降至 950ms，效果显著。

### 仅对必要的位置进行归并

```cpp
void half_merge(float *data, float *recv_data, float *tmp, int len, int recv_len, int tag) {
  if (!tag) { // 若当前进程位置靠左
    for (int i = 0, j = 0, k = 0; k < len;) { // 归并排序
      if ((j == recv_len) || (i < len && (data[i] - recv_data[j]) < EPS)) {
        tmp[k++] = data[i++];
      } else {
        tmp[k++] = recv_data[j++];
      }
    }
  } else { // 若当前进程位置靠右
    ...
  }
}
```

对于 56 进程 $n = 10^8$，从 950ms 降至 780ms，效果显著。

### 二分查找仅传输必要数据

```cpp
int l = 0, r = block_len - 1; // 二分查找必要传输位置
  while (l < r) {
  int mid = (l + r) >> 1;
  if ((tmp[j][mid] - minimum) > EPS) r = mid;
  else l = mid + 1;
}

MPI_Sendrecv(tmp[j] + l, block_len - l, MPI_FLOAT, right, 0,
             recv_data, block_size, MPI_FLOAT, right, 0,
             MPI_COMM_WORLD, &status);
```

对于 56 进程 $n = 10^8$，从 780ms 降至 745ms，效果良好。

## 特定进程运行结果

| 配置   | 运行时间（ms） | 加速比 |
|--------|---------------|--------|
| 1×1    | 12457.163     | 1.00   |
| 1×2    | 6595.442      | 1.89   |
| 1×4    | 3512.590      | 3.55   |
| 1×8    | 2015.089      | 6.18   |
| 1×16   | 1239.289      | 10.06  |
| 2×16   | 1229.790      | 10.13  |