Building

```
mkdir build
cd build
cmake ..
```

Running

```
mkdir output
rm *.mm; make -j8 && mpirun --hostfile hostfile -n 5 ./bin/mpi_gaussian && python view_samples.py && python view_latencies.py
rm *.mm; make -j8 && mpirun --hostfile hostfile -n 5 ./bin/mpi_gaussian_imbalance && python view_samples.py && python view_latencies.py
rm *.mm; make -j8 && mpirun --hostfile hostfile -n 5 ./bin/mpi_lda_sgrld && python view_perplexities_testdata.py && python view_latencies.py
rm *.mm; make -j8 && mpirun --hostfile hostfile -n 5 ./bin/mpi_lda_testdata && python view_perplexities_testdata.py && python view_latencies.py
```
