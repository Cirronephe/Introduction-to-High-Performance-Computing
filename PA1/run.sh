#!/bin/bash

target_program=$1
num_elements=$2
input_file=$3

shift 3

if [ "$num_elements" -lt 1000 ]; then
    num_procs=1
elif [ "$num_elements" -lt 10000 ]; then
    num_procs=2
elif [ "$num_elements" -lt 100000 ]; then
    num_procs=14
elif [ "$num_elements" -lt 1000000 ]; then
    num_procs=28
elif [ "$num_elements" -le 100000000 ]; then
    num_procs=56
else
    num_procs=56
fi

srun -n $num_procs --cpu-bind sockets $target_program $num_elements $input_file $*
