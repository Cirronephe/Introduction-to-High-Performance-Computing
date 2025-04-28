#!/bin/bash

# 你想测试的 UNROLL_N 值列表
for n in 1 2 4 8 16; do
    echo "=============================="
    echo "Building and running with UNROLL_N=$n"
    make clean > /dev/null
    make UNROLL_N=$n
    echo "Running ./main (UNROLL_N=$n)"
    ./main
    echo ""
done