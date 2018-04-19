#!/usr/bin/env bash
make $1 && mpirun --hostfile hostfile -n 5 ./bin/$1 && python view_samples.py && python view_latencies.py
