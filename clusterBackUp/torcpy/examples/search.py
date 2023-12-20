from mpi4py import MPI

def search_array_parallel(arr, val):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    n = len(arr)
    chunk_size = (n + size - 1) // size
    start = rank * chunk_size
    end = min(start + chunk_size, n)

    local_arr = arr[start:end]

    # Initialize the state machine
    state = 0
    found = False
    i = 0
    index = None

    # Run the state machine until a match is found
    while True:
        print("state ->",state, ", i ->", i,", rank ->", rank, found)
        if state == 0:
            if found:
                state = 2
                continue
            else:
                state = 1
            # Check the global flag
            for r in range(0, size):
                print("doing this",rank,r)
                if rank != r:
                    print("doing this2",rank,r, found)
                    found = comm.bcast(found, root=r)
                    print("doing this22",rank,r, found)
                if found:
                    print(rank, found)
                    state = 2
                    break
        elif state == 1:
            # Search the local array
            if i < len(local_arr):
                if local_arr[i] == val:
                    found = True
                    index = i + start
                    # Broadcast the flag to stop searching
                    comm.bcast(found, root=rank)
                    print("found in rank -> ",rank,", index -> ", index, found)
                state = 0
                i += 1
            else:
                state = 2
        elif state == 2:
            # Stop searching
            break

    print("arrived here with rank",rank,index,i,found)
    # Send the index of the first match to the root process
    if rank == 0:
        if not found:
            comm.barrier()
            for r in range(1, size):
                found = comm.bcast(found, root=r)
                if found:
                    return index
        return index
    else:
        comm.Ibarrier()

def parallel_search(arr, key):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    n = len(arr)
    chunk_size = (n + size - 1) // size
    start = rank * chunk_size
    end = min(start + chunk_size, n)

    local_arr = arr[start:end]
    local_index = -1
    for i, val in enumerate(local_arr):
        if val == key:
            local_index = start + i
            break

    index = comm.gather(local_index, root=0)
    if rank == 0:
        for i, val in enumerate(index):
            if val != -1:
                return val

    return -1

if __name__ == '__main__':
    arr = [2, 4, 7, 9, 10, 3, 6, 8, 1, 5]
    key = 7

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        index = search_array_parallel(arr, key)
        if index != -1:
            print("Found key at index:", index)
        else:
            print("Key not found.")
    else:
        search_array_parallel(arr, key)
