import torcpy as torc
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

send_buffer_left = np.zeros(1)
recv_buffer_left = np.zeros(1)
send_buffer_right = np.zeros(1)
recv_buffer_right = np.zeros(1)


def stencil(x):
    # apply a stencil operation on x with weights w
    w = np.array([-1, 0, 1])
    result = []
    print("HERE")
    print(x)
    for i in range(0, 6):
        aux = []
        print(aux)
        if (i == 0 and rank == 0):
            aux = x[0]
        elif (i == 5 and rank == size -1):
            aux = x[5]
        elif (i == 0 and rank > 0):
            send_buffer_left[0] = x[i]
            comm.Isend(send_buffer_left, dest=rank - 1)
            comm.Irecv(recv_buffer_left, source=rank - 1)
            aux = sum([recv_buffer_left[0] * w[0], x[i] * w[1], x[i+1] * w[2]])
        elif (i == 5 and rank < size):
            send_buffer_right[0] = x[i]
            comm.Isend(send_buffer_right, dest=rank + 1)
            comm.Irecv(recv_buffer_right, source=rank + 1)
            aux = sum([x[i-1] * w[0], x[i] * w[1], recv_buffer_right[0] * w[2]])
        else:
            aux = sum([x[i-1] * w[0], x[i] * w[1], x[i+1] * w[2]])
        print("i = ",i, " rank = ", rank, "  ->",aux, "  ->  ", result, "   ->   ", rank)
        result.append(aux)
    return result


def main():
    print("bla1")
    data = [[3,13,7,2,6,9],[0,31,5,18,35,23],[48,78,1,22,43,10]]
    print(data)
    result = torc.map(stencil, data)
    print("bla4")
    print(result)


if __name__ == '__main__':
    torc.start(main)
