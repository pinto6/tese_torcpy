import os
import pstats

# Specify the path to the directory containing the folders
base_path = './cprof'

# Iterate over every folder in the base_path
for folder_name in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder_name)

    # Ensure that it's actually a folder
    if os.path.isdir(folder_path):

        # Iterate over every file in the folder
        for file_name in os.listdir(folder_path):

            # If the file is a .pstats file
            if file_name.endswith('.pstats'):
                file_path = os.path.join(folder_path, file_name)

                # Create a pstats.Stats object
                stats = pstats.Stats(file_path)

                # Sort the statistics by the cumulative time spent in the function
                stats.sort_stats('cumulative')

                # Open a file in write mode
                with open(os.path.join(folder_path, 'output.txt'), 'w') as f:
                    # Redirect the output of print_stats to the file
                    stats.stream = f
                    stats.print_stats()

                # Stop after converting the first .pstats file in the folder
                break
