import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# define stencil size
stencil_size = 3

# create example array
arr = np.arange(30, dtype=np.float64)

# create empty array for output
output = np.zeros_like(arr)

stencilIndexSize = int(len(arr)/size)
stencilIndex = stencilIndexSize*rank
remainder = 0 if (rank != size -1) else len(arr)%size

print("stencil index ",stencilIndex, " rank ",rank, " size ", size)
print(remainder)

# perform stencil calculation
for i in range(stencilIndex, stencilIndex + stencilIndexSize - 1 +remainder):
    local_sum = np.sum(arr[i-stencil_size//2:i+stencil_size//2+1])
    output[i] = local_sum / stencil_size

# communicate results between ranks
if rank == 0:
    recvbuf = np.empty_like(output)
    comm.Reduce(output, recvbuf, op=MPI.SUM, root=0)
    print("Input array:", arr)
    print("Output array:", recvbuf)
else:
    comm.Reduce(output, None, op=MPI.SUM, root=0)
