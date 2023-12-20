# Command
mpirun --hostfile $OAR_NODEFILE -n 16 --mca btl_tcp_if_include bond0 tests.py

no .ssh/config:
StrictHostKeyChecking no


# READ ME
TorcPy is on torc_py/torcpy

benchmark.sh is a script to run in the cluster.

tests.py is my test to the torcPy with my extension (search and stencil)

searchNoTorc.py is the test to the search pattern using only MPI.

searchTorcMap.py is the test to the search pattern using the original TorcPy.

inside tempos there is a script to create graphs and there is the times of the benchmarks
