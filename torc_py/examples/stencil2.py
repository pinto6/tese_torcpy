import numpy as np
from mpi4py import MPI

def moving_average_parallel(data):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    N = len(data)
    chunk_size = N // size
    remainder = N % size

    # Determine size of each chunk
    chunk_sizes = np.ones(size, dtype=int) * chunk_size
    chunk_sizes[:remainder] += 1

    # Determine the starting and ending indices for each chunk
    starts = np.zeros(size, dtype=int)
    starts[1:] = np.cumsum(chunk_sizes)[:-1]
    ends = starts + chunk_sizes

    # Determine the starting and ending indices for the overlap regions
    overlap_starts = np.maximum(starts - 1, 0)
    overlap_ends = np.minimum(ends + 1, N)

    # Initialize arrays to hold the results
    chunk_result = np.zeros(chunk_sizes[rank] + 2, dtype=float)
    result = None
    if rank == 0:
        result = np.zeros(N, dtype=float)

    # Compute the local moving average
    for i in range(starts[rank], ends[rank]):
        chunk_result[i - starts[rank] + 1] = np.mean(data[overlap_starts[rank]+1:overlap_ends[rank]-1])

    # Gather the results
    if rank == 0:
        comm.Gather(MPI.IN_PLACE, result, root=0)
    else:
        comm.Gather(chunk_result[1:-1].tobytes(), None, root=0)

    # Return the result
    if rank == 0:
        return result
    else:
        return None


data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
result = moving_average_parallel(data)
print(result)