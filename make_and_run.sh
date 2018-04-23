#!/usr/bin/env bash
rm *.mm \
    ; cd build \
    && make -j4 $1 \
    && cd ../ \
    && mpirun -hostfile hostfile -n 2 ./build/bin/$1 \
    && python view_samples.py \
    && python view_latencies.py
