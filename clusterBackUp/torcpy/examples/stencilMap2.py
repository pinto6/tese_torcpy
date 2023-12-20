import torcpy as torc
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()



def stencil(my_tuple):
    data = my_tuple[0]
    stencilFunction = my_tuple[1]
    radius = my_tuple[2]

    send_buffer_left = np.zeros(radius)
    recv_buffer_left = np.zeros(radius)
    send_buffer_right = np.zeros(radius)
    recv_buffer_right = np.zeros(radius)

    dataSize= len(data)

    #it is only working for filters of 3 elements
    result = []
    for i in range(dataSize):
        #edge cases, what to do? should stay same or add 0 to the left or right?
        if ((i < radius and rank == 0) or (i > dataSize-1-radius and rank == size -1)):
            aux = data[i]
        elif (i < radius and rank > 0):
            send_buffer_left = sendBuffer(data[:radius], send_buffer_left)
            comm.Isend(send_buffer_left, dest=rank - 1, tag= dataSize-(radius-i) + (rank-1)*dataSize)
            comm.Recv(recv_buffer_left, source=rank - 1, tag= i + rank*dataSize)
            aux = stencilFunction(receiveBuffer(recv_buffer_left[-radius + i:],get_neighbors(data,i,radius)))
        elif (i > dataSize-1-radius and rank < size-1):
            send_buffer_right = sendBuffer(data[-radius:],send_buffer_right)
            comm.Isend(send_buffer_right, dest=rank + 1, tag=radius - (dataSize-i) + (rank+1)*dataSize)
            comm.Recv(recv_buffer_right, source=rank + 1, tag=i + rank*dataSize)
            aux = stencilFunction(receiveBuffer(get_neighbors(data,i,radius),recv_buffer_right[:radius - (dataSize -i -1)]))
        else:
            aux = stencilFunction(get_neighbors(data,i,radius))
        result.append(aux)
    return result


def sendBuffer(arr,send_buffer):
    for i in range(len(send_buffer)):
        send_buffer[i] = arr[i]
    return send_buffer

def receiveBuffer(arr,recv):
    size1 = len(arr)
    size2 = len(recv)
    receive = np.zeros(size1+size2)
    for i in range(size1+size2):
        if( i < size1):
            receive[i] = arr[i]
        else:
            receive[i] = recv[i-size1]
    return receive




def get_neighbors(arr, index, radius):
    start = max(0, index - radius)
    end = min(len(arr), index + radius + 1)
    return arr[start:end]

def stencilFilter(data):
    stencilFilter = np.array([0,-1, 0, 1,0])
    result = sum([x * y for x, y in zip(data, stencilFilter)])
    return result


def main():
    print("init")
    data = [3,13,7,2,6,9,1,31,5,18,35,23,48,78,1,22,43,10]
    data = convertArrayToMapArgument(data,3,2)
    print(data)
    result = torc.map(stencil, data)
    print("end")
    print(result)

def convertArrayToMapArgument(data,nodes,radius):
    subarrays = []
    size = len(data)
    start = 0
    step = size // nodes
    end = step
    for i in range(nodes):
        if i == nodes-1:
            end = size  # last sub-array goes until the end
        subarrays.append((data[start:end], stencilFilter, radius))
        start = end
        end += step
    return subarrays


if __name__ == '__main__':
    torc.start(main)
