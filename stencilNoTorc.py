#!/usr/bin/env python3

from mpi4py import MPI
import numpy as np
import time


comm = MPI.COMM_WORLD

def stencilOperation(sub_data, radius, filter_func, originalShape, worker_chunk_size):
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()

    # Create buffers for exchanging data
    send_buffer_top = np.zeros(sub_data.shape[1])
    send_buffer_bottom = np.zeros(sub_data.shape[1])
    recv_buffer_top = np.zeros(sub_data.shape[1])
    recv_buffer_bottom = np.zeros(sub_data.shape[1])


    # Perform stencil computation
    if rank > 0:
        send_buffer_top = sub_data[radius:2*radius, :]
        comm.Isend(send_buffer_top, dest=rank - 1)
        comm.Recv(recv_buffer_top, source=rank - 1)
        sub_data[:radius, :] = recv_buffer_top

    if rank < size - 1:
        send_buffer_bottom = sub_data[-2*radius:-radius, :]
        comm.Isend(send_buffer_bottom, dest=rank + 1)
        comm.Recv(recv_buffer_bottom, source=rank + 1)
        sub_data[-radius:, :] = recv_buffer_bottom

    
    # Apply filter function to each index of sub_data
    result = np.zeros((sub_data.shape[0] - 2 * radius, sub_data.shape[1] - 2 * radius))
    start = radius
    end = sub_data.shape[0] - radius

    for i in range(start, end):
        for j in range(radius, sub_data.shape[1] - radius):
            window = sub_data[i - radius : i + radius + 1, j - radius : j + radius + 1]
            aux = filter_func(window)
            result[i - radius, j - radius] = aux


    # Update sub_data with the computed results
    sub_data = result

    # Gather the results from all processes
    gathered_data = None
    if rank == 0:
        gathered_data = np.zeros((size*sub_data.shape[0], sub_data.shape[1]))
    comm.Gather(sub_data, gathered_data, root=0)

    return gathered_data


def stencil2DSubmited(data,radius,originalShape,function):
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    num_rows = (originalShape[0] + size -1) // size
    # Scatter the data to all processes
    if rank == 0:
        local_arr = comm.scatter([data[i:i+num_rows, :] for i in range(0, size * num_rows, num_rows)], root=0)
    else:
        local_arr = comm.scatter(data,root=0)
    

    worker_chunk_size = (num_rows + size - 1) // size

    if local_arr.shape[0] != num_rows:
        aux_arr = np.zeros((num_rows,originalShape[1]))
        aux_arr[:local_arr.shape[0],:] = local_arr
        local_arr = aux_arr

    #local_arr = local_arr[worker_local_id()*worker_chunk_size:(worker_local_id()+1)*worker_chunk_size,:]

    sub_data = np.zeros((local_arr.shape[0] + 2 * radius, local_arr.shape[1] + 2* radius))


    if rank < size - 1:
        sub_data[radius:-radius, radius:-radius] = local_arr
    else:
        # Last rank handles the remaining rows
        sub_data[radius:-radius, radius:-radius] = local_arr


    # Perform stencil computation
    result = stencilOperation(sub_data, radius=radius, filter_func=function, originalShape=originalShape, worker_chunk_size=worker_chunk_size)

    if rank == 0:
        result = (rank,result[:originalShape[0],:])
        return result

def filter_function(sub_array):
    filter = np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]])
    result = None
    radius = filter.shape[0] // 2
    result = np.sum(sub_array * filter)
    return result

def create_2d_array(rows, cols):
    # Create an array of arrays filled with 1
    result_array = np.ones((rows, cols), dtype=int)

    return result_array


# Test the function
if __name__ == "__main__":
    data = create_2d_array(10000,10000)

    radius = 1
    rank = comm.Get_rank()

    ts = 0
    if rank == 0:
        ts = time.time()
        toRet = stencil2DSubmited(data,radius,data.shape,function=filter_function)
        ts = time.time() - ts
        print("ended execution","with time",ts)
        print("to return", toRet)
    else:
        toRet = stencil2DSubmited(None,radius,data.shape,function=filter_function)
    


    
