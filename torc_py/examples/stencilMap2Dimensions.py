import torcpy as torc
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def stencil2D(my_tuple):
    data = my_tuple[0]
    stencilFunction = my_tuple[1]
    radius = my_tuple[2]
    
    result = []
    for i in range(radius, data.shape[0]-radius):
        for j in range(radius, data.shape[1]-radius):
            result.append(stencilFunction(data[i-radius:i+radius+1,j-radius:j+radius+1]))
    return result


def stencilFilter(data):
    stencilFilter = np.array([[1, 0, 1],[0,1,0],[1,0,1]])
    result = np.zeros(3)
    for i in range(0,3):
        result[i] = sum([x * y for x, y in zip(data[i,:], stencilFilter[i,:])])
    return sum(result)


def main():
    print("init")
    data = np.array([
        [3, 5, 8, 5, 0, 9, 4, 7, 4],
        [8, 4, 4, 4, 4, 2, 9, 1, 7],
        [2, 2, 1, 6, 7, 1, 1, 8, 9],
        [1, 5, 8, 7, 3, 2, 1, 6, 8],
        [6, 9, 5, 6, 7, 3, 2, 2, 1],
        [8, 9, 0, 7, 6, 2, 5, 6, 7],
        [6, 1, 9, 9, 7, 4, 8, 1, 4],
        [4, 4, 2, 2, 3, 2, 5, 1, 8],
        [5, 1, 5, 3, 5, 1, 9, 9, 8]
    ])
    data = stencil2D((data,stencilFilter, 1))
    #data = convertArrayToMapArgument(data,3,2)
    print(data)
    #result = torc.map(stencil, data)
    print("end")
    #print(result)

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
