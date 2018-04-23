#!/usr/bin/env bash
cd build \
    && rm *.mm\
    ; make -j4 $1 \
    && cd ../ \
    && mpirun -hostfile hostfile -n 4 ./build/bin/$1 \
    && python view_samples.py\
    ; python view_latencies.py
