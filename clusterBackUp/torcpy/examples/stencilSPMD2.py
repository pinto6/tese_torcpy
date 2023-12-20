import numpy as np
import torcpy as torc
from mpi4py import MPI

def stencil(A, start_index, end_index):
    """
    A function that performs a simple stencil operation on a numpy array.
    """
    B = np.zeros_like(A)
    for i in range(start_index, end_index):
        if i > 0 and i < A.shape[0] - 1:
            B[i] = (A[i-1] + A[i] + A[i+1]) / 3
        else:
            B[i] = A[i]
    return B

def work():
    N = 30000
    A = np.zeros(N, dtype=np.float64)
    for i in range(0, N):
        A[i] = 100 * i

    # Initialize MPI communicator and get rank and size
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    chunk_size = int(N / size)
    start_index = rank * chunk_size
    end_index = (rank + 1) * chunk_size if rank < size - 1 else N

    # Apply the stencil operation to the local portion of the array
    B = stencil(A, start_index, end_index)

    # Communicate the boundary values between neighboring processes
    if rank > 0:
        comm.send(B[start_index], dest=rank-1)
    if rank < size - 1:
        comm.send(B[end_index-1], dest=rank+1)
    if rank > 0:
        A[start_index-1] = comm.recv(source=rank-1)
    if rank < size - 1:
        A[end_index] = comm.recv(source=rank+1)

    # Combine the results from all processes into a single array on process 0
    if rank == 0:
        final_result = B
        for i in range(1, size):
            local_result = comm.recv(source=i)
            final_result = np.concatenate([final_result, local_result])
    else:
        comm.send(B, dest=0)

    # Print the final result from process 0
    if rank == 0:
        print("Final result:", final_result)

def main():
    torc.spmd(work)

if __name__ == '__main__':
    torc.start(main)
