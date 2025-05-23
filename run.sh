#!/bin/bash

# Set the GPU ID you want to monitor
GPU_ID=5
# Set a threshold for GPU memory usage that indicates the GPU is idle (e.g., 100 MB)
IDLE_MEMORY_THRESHOLD=1000
# Set the time (in seconds) between checks
CHECK_INTERVAL=10

# Function to check if the GPU memory usage is below the threshold
is_gpu_memory_idle() {
    # Use nvidia-smi to get the memory usage of the GPU
    MEMORY_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU_ID)
    # Check if the memory usage is below the threshold
    if [ "$MEMORY_USED" -lt "$IDLE_MEMORY_THRESHOLD" ]; then
        return 0 # GPU memory is idle
    else
        return 1 # GPU memory is still in use
    fi
}

# Monitor the GPU memory until it becomes idle
echo "Monitoring GPU $GPU_ID for idle memory state..."
while true; do
    if is_gpu_memory_idle; then
        echo "GPU $GPU_ID memory is idle. Running the next program..."
        # Run your next program here
        bash llama3_babilong.sh  # Replace with your command
        break
    else
        echo "GPU $GPU_ID memory in use: $MEMORY_USED MB. Checking again in $CHECK_INTERVAL seconds..."
    fi
    # Wait before checking again
    sleep $CHECK_INTERVAL
done
