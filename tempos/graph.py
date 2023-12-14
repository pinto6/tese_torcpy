import os
import matplotlib.pyplot as plt

# specify the path to your text files
path = './charmander'

# get a list of all text files in the specified directory
files = [f for f in os.listdir(path) if f.endswith('.txt')]

# iterate over each file
for file in files:
    # create a new figure for each file
    plt.figure()

    # open the file
    with open(os.path.join(path, file), 'r') as f:
        # read the lines of the file
        lines = f.readlines()

        # set the title of the plot to the first line of the file
        plt.title(lines[0].strip())

        # parse the data
        data = [line.strip().split(' - ') for line in lines[2:]]
        iterations = [int(d[0]) for d in data]
        times = [float(d[1]) for d in data]

        # plot the data
        plt.plot(iterations, times, label=file)

        # label the axes
        plt.xlabel('Nodes')
        plt.ylabel('Time (ms)')

        # add a legend
        plt.legend()

        # save the plot as a PNG file with the same name as the text file
        plt.savefig(f'{file}.png')

    # close the plot
    plt.close()
