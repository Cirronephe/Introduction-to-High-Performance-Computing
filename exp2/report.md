# MPI Allreduce 小作业 实验报告 

颜子俊 2023010828

## 实现算法简述

首先将 `n` 分为 `comm_sz` 组。

第一阶段，共 `comm_sz - 1` 步。每一步用 `recvbuf` 的第 `(my_rank - 1 - i + comm_sz) % comm_sz` 块从前一个进程非阻塞接收，将 `sendbuf` 的第 `(my_rank - i + comm_sz) % comm_sz` 块非阻塞发送至后一个进程。同步后，`sendbuf` 的第 `(my_rank - 1 - i + comm_sz) % comm_sz` 块逐位累加 `recvbuf` 的同位块。

第二阶段，共 `(comm_sz - 1) + 1 = comm_sz` 步。迭代接着第一阶段，接收发送相同。同步后，`sendbuf` 的第 `(my_rank - 1 - i + comm_sz) % comm_sz` 块逐位赋值为 `recvbuf` 的同位块。多迭代一步让 `recvbuf` 中所有块都变为最终答案。

## 通信时间
```
Correct.
MPI_Allreduce:   2597.86 ms.
Naive_Allreduce: 5035.57 ms.
Ring_Allreduce:  2085.99 ms.
```