# CUDA 优化 小作业 实验报告 

颜子俊 2023010828

## 测量结果

<img src="C:\Users\yanzj\AppData\Roaming\Typora\typora-user-images\image-20250423160651493.png" alt="image-20250423160651493" style="zoom:67%;" />

<img src="C:\Users\yanzj\AppData\Roaming\Typora\typora-user-images\image-20250423160708301.png" alt="image-20250423160708301" style="zoom:67%;" />

## 问题回答

### Global memory

- 合并访存。

- Stride 越大，访问越不连续，合并效率越低，带宽越低。

- 缓存，对于 stride 较小的访问缓存命中率较高；内存对齐，若 stride 选择不恰当会增加额外的访问次数。

### Shared memory

- Bank Conflict。
- Bitwidth = 2, 8 都存在“平台”。对于 bitwidth = 2，stride 1 $\to$ 2 是平台，因为一个 warp 共 32 线程，stride = 1, 2 时都不出现 bank conflict，带宽变化微小；对于 bitwidth = 8，stride 16 $\to$ 32 是平台，因为 stride = 16 时已经达到 32-way bank conflict。
