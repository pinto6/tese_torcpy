#!/usr/bin/env python3

from mpi4py import MPI
import torcpy as torc
import numpy as np
import time


comm = MPI.COMM_WORLD

def stencilOperation(sub_data, radius, filter_func):

    size = torc.num_nodes()
    rank = torc.node_id()

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


def stencil2DSubmited(data):
    originalShape=(10000,10000)
    radius = 1
    function=filter_function
    size = torc.num_nodes()
    rank = torc.node_id()
    num_rows = (originalShape[0] + size -1) // size    

    if data.shape[0] != num_rows:
        aux_arr = np.zeros((num_rows,originalShape[1]))
        aux_arr[:data.shape[0],:] = data
        data = aux_arr


    sub_data = np.zeros((data.shape[0] + 2 * radius, data.shape[1] + 2* radius))


    sub_data[radius:-radius, radius:-radius] = data


    # Perform stencil computation
    result = stencilOperation(sub_data, radius=radius, filter_func=function)


    if rank == 0:
        result = (rank,result[:originalShape[0],:])
        return result
    return None

def filter_function(sub_array):
    filter = np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]])
    result = np.sum(sub_array * filter)
    return result

def create_2d_array(rows, cols):
    # Create an array of arrays filled with 1
    result_array = np.ones((rows, cols), dtype=int)

    return result_array

def subArray(arr,nodes):
    length = arr.shape[0]
    chunk_size = (length + nodes - 1) // nodes
    sub_arr= [arr[i:i+chunk_size,:] for i in range(0, length, chunk_size)]
    return sub_arr



def main():
    
    size = torc.num_nodes()
    
    rank = torc.node_id()



    data = create_2d_array(10000,10000)

    result = None
    ts = time.time()
    sub_arr= subArray(data,size)
    results = torc.map(stencil2DSubmited,sub_arr)

    for r in results:
        if r != None:
            result= r
            break

    #gathered_data = None
    #if rank == 0:
    #    gathered_data = np.zeros((data.shape[0], data.shape[1]))
    
    #last_chunk_size = (data.shape[0]-((size-1)*chunk_size))
    #for r in results:
    #    if r[0] < size - 1:
    #        gathered_data[r[0]*chunk_size:(r[0]+1)*chunk_size,:] = r[1]
    #    else:
    #        gathered_data[r[0]*chunk_size:data.shape[0],:] = r[1][0:last_chunk_size,:]

    ts = time.time() - ts
    print("RESULT IS ",result)
    print("ended execution","with time",ts)
    
if __name__ == '__main__':
    torc.start(main)