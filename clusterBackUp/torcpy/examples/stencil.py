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

# determine local stencil indices
stencilIndexSize = int(len(arr) / size)
stencilIndex = stencilIndexSize * rank
remainder = 0 if (rank != size - 1) else len(arr) % size -1

# send and receive buffers
send_buffer_left = np.zeros(1)
recv_buffer_left = np.zeros(1)
send_buffer_right = np.zeros(1)
recv_buffer_right = np.zeros(1)

# perform stencil calculation
for i in range(stencilIndex, stencilIndex + stencilIndexSize + remainder):
    local_sum = np.sum(arr[i - stencil_size // 2:i + stencil_size // 2 + 1])

    # send edge value to the left
    if i == stencilIndex:
        if rank > 0:
            send_buffer_left[0] = arr[i]
            comm.Isend(send_buffer_left, dest=rank - 1)
            comm.Irecv(recv_buffer_left, source=rank - 1)
            local_sum += recv_buffer_left[0]

    # send edge value to the right
    if i == stencilIndex + stencilIndexSize - 1:
        print (rank)
        if rank < size - 1:
            send_buffer_right[0] = arr[i]
            comm.Isend(send_buffer_right, dest=rank + 1)
            comm.Irecv(recv_buffer_right, source=rank + 1)
            local_sum += recv_buffer_right[0]

    output[i] = local_sum / stencil_size

# communicate results between ranks
if rank == 0:
    recvbuf = np.empty_like(output)
    comm.Reduce(output, recvbuf, op=MPI.SUM, root=0)
    print("Input array:", arr)
    print("Output array:", recvbuf)
else:
    comm.Reduce(output, None, op=MPI.SUM, root=0)
