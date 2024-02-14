#!/usr/bin/env python3

import torcpy as torc
import numpy as np
import time
from mpi4py import MPI
#import cProfile

def create_2d_array(rows, cols):
    # Create an array of arrays filled with 1
    result_array = np.ones((rows, cols), dtype=int)

    return result_array


def filter_function(sub_array):
    filter = np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]])
    result = None
    radius = filter.shape[0] // 2
    result = np.sum(sub_array * filter)
    return result
 
def stencil():
    data = create_2d_array(10000,10000)
    print(data.shape)

    radius = 1
    ts = time.time()
    toRet = torc.stencil2D(data,radius,function=filter_function)
    ts = time.time() - ts
    
    print("to return", toRet)
    print("TIME",ts)

#def torcStart():
#    torc.start(stencil)
        
if __name__ == "__main__":
    torc.start(stencil)
    #rank = MPI.COMM_WORLD.Get_rank()
    #cProfile.run('torcStart()', 'cprof/stencil/output{}.pstats'.format(rank))
