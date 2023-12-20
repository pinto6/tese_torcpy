import numpy as np
import torcpy as torc
from mpi4py import MPI

# Define array size and stencil size
N = 10
stencil_size = 1

# Initialize array A
A = np.zeros(N, dtype=np.float64)
for i in range(N):
    A[i] = i

# Create an empty array for the stencil result
result = np.zeros_like(A)

# Define the work function to perform the stencil calculation
def stencil_work():
    global A, result
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Determine local stencil indices
    stencilIndexSize = int(N / size)
    stencilIndex = stencilIndexSize * rank
    remainder = 0 if (rank != size - 1) else N % size

    # Perform stencil calculation
    for i in range(stencilIndex + remainder, stencilIndex + stencilIndexSize):
        neighbor_sum = A[max(0, i - stencil_size):min(N, i + stencil_size + 1)]
        result[i] = np.mean(neighbor_sum)

    # Print the local result
    print("Node {} result: {}".format(rank, result[stencilIndex:stencilIndex+stencilIndexSize]))
def main():
    # Perform stencil calculation in parallel using torc.spmd
    torc.spmd(stencil_work)

if __name__ == '__main__':
    torc.start(main)
