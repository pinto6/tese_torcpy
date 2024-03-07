#!/usr/bin/env python3

from mpi4py import MPI
import numpy as np
import time
from mpi4py import MPI
#import cProfile


comm = MPI.COMM_WORLD

def fun(value):
    #time.sleep(0.1)
    #return value == 20
    start_time = time.time()  # Record the starting time
    total_sum = 0
    for i in range(1, 1000000):
        total_sum += i

        # Check if 100 milliseconds have elapsed
        elapsed_time = time.time() - start_time
        if elapsed_time >= 0.00001:
            break

    return value == 20

def transform(value):
    return value*10 if value !=None else None 

def broadcast(comm,rank,size,value):
    if rank != 0:
        return
    i= 1
    while i< size:
        comm.Isend(value,dest = i)
        i += 1

def search_array_parallel(arr, length, checkFunction, transformFunc=lambda a: a):
    size = comm.Get_size()
    rank = comm.Get_rank()


    chunk_size = (length + size - 1) // size
    start = rank * chunk_size



    # Scatter the data to all processes
    if rank == 0:
        local_arr = comm.scatter([arr[i:i+chunk_size] for i in range(0, length, chunk_size)], root=0)
    else:
        local_arr = comm.scatter(arr,root=0)
    
    #local_arr = arr[start:end]

    # Pad the subarray in the last rank with None values
    if rank == size - 1:
        local_arr += [None] * (chunk_size - len(local_arr))

    # Initialize the state machine
    state = 0
    found = False
    i = 0
    index = None
    originalValue = None
    transformedValue = None

    # Shared variable to signal other processes
    found_signal = bytearray(1)
    found_signal[0] = int(found)


    while True:
        if( i%1000 == 0 and rank == 31):
            print("state ->",state, ", i ->", i,", rank ->", rank, found)        
        if state == 0:
            # Check if any process has broadcasted the signal
            if comm.Iprobe(source=MPI.ANY_SOURCE):
                comm.Recv(found_signal, source=MPI.ANY_SOURCE)
            if found or found_signal[0] == 1:
                # Another process found the value, stop work
                broadcast(comm,rank,size,found_signal)
                break
            state = 1
        elif state == 1:
            if i < len(local_arr):
                transformedValue = transformFunc(local_arr[i])
                if checkFunction(transformedValue):
                        found = True
                        originalValue = local_arr[i]
                        index = i + start

                        # Broadcast the found signal to other processes
                        found_signal[0] = 1
                        if rank != 0:
                            comm.Isend(found_signal,dest = 0)
                        #comm.Bcast(found_signal, root=rank)
                i += 1
                state = 0
            else:
                break

    # Gather the results to rank 0
    results = comm.gather((found, index, originalValue, transformedValue), root=0)

    if rank == 0:
        # Process the results from all ranks
        for r in results:
            if r[0]:
                print("Value found at index:", r[1], "with original value", r[2], "and transformed value", r[3])
                return r

def main():
    arr = [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3
    ,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3
    ,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3
    ,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3
    ,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3
    ,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3
    ,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3
    ,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3
    ,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3
    ,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]


    arr = arr + arr + arr + arr + arr
    arr = arr + arr
    arr = arr + arr + arr + arr + arr
    arr = arr + arr
    arr = arr + arr + arr + arr + arr
    arr = arr + arr
    arr = arr + arr + arr + arr + arr
    arr = arr + arr
    arr = arr + arr + arr + arr + arr
    arr = arr + arr
    
    from mpi4py import MPI
    #numNodes = MPI.COMM_WORLD.Get_size()
    #index = int(len(arr)/2)
    #subArrayLen = int(len(arr)/numNodes)
    #index = int(subArrayLen * (numNodes/2) + (subArrayLen/2))
    #if numNodes == 1:
    #    index = 80000000
    #print("index array is", index)
    #arr[index] = 2
    arr[-1] = 2

    #arr = [4, 2, 7, 1, 9, 5, 8]
    
    rank = comm.Get_rank()

    if rank == 0:
        print(len(arr))
        ts = time.time()
        search_array_parallel(arr, len(arr), checkFunction=fun, transformFunc = transform)
        ts = time.time() - ts
        print("ended execution","with time",ts)
    else:
        search_array_parallel(None, len(arr), checkFunction=fun, transformFunc = transform)

# Test the function
if __name__ == "__main__":
    main()
    #rank = MPI.COMM_WORLD.Get_rank()
    #cProfile.run('main()', 'cprof/searchNoTorc/output{}.pstats'.format(rank))
    


    
