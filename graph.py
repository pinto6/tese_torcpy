import matplotlib.pyplot as plt

# Data
bulbasaur = [1, 2, 4, 8, 16, 32]
time_sleep = [269.0885570049286, 132.850683927536, 68.19914937019348, 38.6680212020874, 22.013906955718994, 52.046875]
no_time_sleep_1 = [257.9383535385132, 132.83798336982727, 71.68426656723022, 37.21051287651062, 21.339755058288574, 50.87347769737244]
no_time_sleep_2 = [263.3126275539398, 131.93462920188904, 67.99005722999573, 37.50864315032959, 21.803712606430054, 51.83280348777771]

# Create subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plot the data
axs[0].plot(bulbasaur, time_sleep, marker='o', label='With time.sleep')
axs[0].set_title('With time.sleep')
axs[0].set_xlabel('Number of MPI')
axs[0].set_ylabel('Time (seconds)')
axs[0].legend()

axs[1].plot(bulbasaur, no_time_sleep_1, marker='o', label='Without time.sleep - 1 bulbasaur')
axs[1].set_title('Without time.sleep - 1 bulbasaur')
axs[1].set_xlabel('Number of MPI')
axs[1].set_ylabel('Time (seconds)')
axs[1].legend()

axs[2].plot(bulbasaur, no_time_sleep_2, marker='o', label='Without time.sleep - 2 bulbasaur')
axs[2].set_title('Without time.sleep - 2 bulbasaur')
axs[2].set_xlabel('Number of MPI')
axs[2].set_ylabel('Time (seconds)')
axs[2].legend()

# Save the plots to a file (in PNG format)
plt.tight_layout()
plt.savefig('graphs.png')

# Show the plots
plt.show()
