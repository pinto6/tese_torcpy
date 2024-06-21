#!/bin/bash

# Check if the output file name is provided as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <output_file>"
    exit 1
fi

# Output file to save results
OUTPUT_FILE="$1"

# Define the command template
CMD="mpirun --hostfile \$OAR_NODEFILE -n NUMBERTHREADS --mca btl_tcp_if_include bond0 searchTorcMap.py"

# Specify the range of thread counts you want to test
THREAD_COUNTS=(16 32 33 64 65 96)

# Output file to save results
OUTPUT_FILE="benchmarks/V2/$1"

# Clear previous results
echo -n "" > "$OUTPUT_FILE"

# Loop through each thread count and run the command
for NUM_THREADS in "${THREAD_COUNTS[@]}"; do
    # Replace NUMBERTHREADS with the current thread count
    CURRENT_CMD="${CMD/NUMBERTHREADS/$NUM_THREADS}"

    echo "run started with $NUM_THREADS at $(date +'%Y-%m-%d %H:%M:%S')" 
    # Run the command and capture the output
    RESULT=$(eval "$CURRENT_CMD")
    echo "run stoped with $NUM_THREADS at $(date +'%Y-%m-%d %H:%M:%S')"

    # Save the result to the output file
    echo "Thread Count: $NUM_THREADS" >> "$OUTPUT_FILE"
    echo "$RESULT" >> "$OUTPUT_FILE"
    echo "----------------------------------------" >> "$OUTPUT_FILE"
done

echo "Benchmarking complete. Results saved to $OUTPUT_FILE"
