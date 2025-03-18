<center><font size="5">HW0</font></center>

<center><font size="3">颜子俊 计35 2023010828</font></center>

## 代码

### `openmp_pow`

```cpp
void pow_a(int *a, int *b, int n, int m) {
    // TODO: 使用 omp parallel for 并行这个循环
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        int x = 1;
        for (int j = 0; j < m; j++)
            x *= a[i];
        b[i] = x;
    }
}
```

### `mpi_pow`

```cpp
void pow_a(int *a, int *b, int n, int m, int comm_sz /* 总进程数 */) {
    // TODO: 对这个进程拥有的数据计算 b[i] = a[i]^m
    int local_n = n / comm_sz;
    for (int i = 0; i < local_n; i++) {
        int x = 1;
        for (int j = 0; j < m; j++)
            x *= a[i];
        b[i] = x;
    }
}
```

## `openmp` 并行加速效果

```
openmp_pow: n = 112000, m = 100000, thread_count = 1
Congratulations!
Time Cost: 14018828 us

openmp_pow: n = 112000, m = 100000, thread_count = 7
Congratulations!
Time Cost: 2017711 us

openmp_pow: n = 112000, m = 100000, thread_count = 14
Congratulations!
Time Cost: 1015406 us

openmp_pow: n = 112000, m = 100000, thread_count = 28
Congratulations!
Time Cost: 515554 us
```

7，14，28 线程加速比分别为 6.948，13.806，27.192。

## `MPI` 并行加速效果

```
mpi_pow: n = 112000, m = 100000, process_count = 1
Congratulations!
Time Cost: 14015328 us

mpi_pow: n = 112000, m = 100000, process_count = 7
Congratulations!
Time Cost: 2013379 us

mpi_pow: n = 112000, m = 100000, process_count = 14
Congratulations!
Time Cost: 1018854 us

mpi_pow: n = 112000, m = 100000, process_count = 28
Congratulations!
Time Cost: 501812 us

mpi_pow: n = 112000, m = 100000, process_count = 56
Congratulations!
Time Cost: 362856 us
```

加速比分别为 6.961，13.756，27.929，38.625。