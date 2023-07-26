from mpi4py import MPI
import numpy as np

# Define the filter function
def filter_function(sub_array):
    filter = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    result = None
    radius = filter.shape[0] // 2
    for i in range(radius, sub_array.shape[0] - radius):
        for j in range(radius, sub_array.shape[1] - radius):
            window = sub_array[i - radius : i + radius + 1, j - radius : j + radius + 1]
            result = np.sum(window * filter)
    return result

# Function to apply stencil operation
def stencilOperation(sub_data, radius, filter_func, originalShape, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()

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
    for i in range(radius, sub_data.shape[0] - radius):
        for j in range(radius, sub_data.shape[1] - radius):
            window = sub_data[i - radius : i + radius + 1, j - radius : j + radius + 1]
            aux = filter_func(window)
            result[i - radius, j - radius] = aux


    # Update sub_data with the computed results
    sub_data = result

    print("rank",rank,"sd",sub_data)

    # Gather the results from all processes
    gathered_data = None
    if rank == 0:
        gathered_data = np.zeros(originalShape)
    comm.Gather(sub_data, gathered_data, root=0)

    return gathered_data

def stencil(data,radius,originalShape,function):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()


    # Scatter the data to all processes
    if rank == 0:
        num_rows = (data.shape[0] + size -1) // size
        local_arr = comm.scatter([data[i:i+num_rows, :] for i in range(0, size * num_rows, num_rows)], root=0)
    else:
        local_arr = comm.scatter(data,root=0)
        num_rows = local_arr.shape[0]
    

    sub_data = np.zeros((local_arr.shape[0] + 2 * radius, local_arr.shape[1] + 2* radius))


    if rank < size - 1:
        sub_data[radius:-radius, radius:-radius] = local_arr
    else:
        # Last rank handles the remaining rows
        sub_data[radius:-radius, radius:-radius] = local_arr

    # Perform stencil computation
    result = stencilOperation(sub_data, radius=radius, filter_func=function, originalShape=originalShape,comm=comm)

    if rank == 0:
        print("Result:")
        print(result)


# Example usage
if __name__ == "__main__":

    #data = np.random.rand(10, 10)
    data = np.array([[1, 2, 3, 4],
                     [5, 6, 7, 8],
                     [9, 10, 11, 12],
                     [13, 14, 15, 16]])
    radius = 1

    #stencil(data,radius,function=filter_function)
    comm = MPI.COMM_WORLD
    
    rank = comm.Get_rank()
    if rank == 0:
        stencil(data,radius,data.shape,function=filter_function)
    else:
        stencil(None,radius,data.shape,function=filter_function)

