# OpenMP 并行 for 循环 小作业 实验报告 

颜子俊 2023010828

## 线程数

选择与逻辑核心数相同为 28。

## 补全的指导语句以及测量结果

```cpp
#pragma omp parallel for num_threads(28) schedule(static)
```

### `static`

Sort uniform parts: 68.4836 ms  
Sort random parts: 190.556 ms

### `dynamic`

Sort uniform parts: 84.6174 ms  
Sort random parts: 167.076 ms

### `guided`

Sort uniform parts: 69.3569 ms  
Sort random parts: 163.199 ms

## 原因分析

对于第一组测例，`static` < `guided` < `dynamic`。由于排序区间均匀且区间数目较多，负载均衡已经较好，动态的分配方式必要性较低，反而带来过多分配开销。

对于第二组测例，`guided` < `dynamic` < `static`。由于排序区间长度随机且区间数较少，动态分配方式可以较好地改善负载均衡，且不会带来过多分配开销。