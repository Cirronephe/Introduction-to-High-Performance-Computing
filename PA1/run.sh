#!/bin/bash

target_program=$1
num_elements=$2
input_file=$3

shift 3

if [ "$num_elements" -le 100000 ]; then
    num_procs=1
elif [ "$num_elements" -le 1000000 ]; then
    num_procs=4
elif [ "$num_elements" -le 10000000 ]; then
    num_procs=16
else
    num_procs=56
fi

srun -n $num_procs $target_program $num_elements $input_file $*
