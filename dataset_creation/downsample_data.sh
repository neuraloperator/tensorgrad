#!/bin/bash
#SBATCH --job-name=my_gh_dev_job
#SBATCH --partition=gh-dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=1:00:00
#SBATCH -o output_files/mps_job.%j.out
#SBATCH -e output_files/mps_job.%j.err
#SBATCH --account=NAIRR240304

# Create output directory if it doesn't exist
mkdir -p output_files

# Print hostname and show GPU info
echo "Running on $(hostname):"
nvidia-smi

# Load required modules
module load gcc cuda
module load python3  # Ensure python3 module is loaded

# Activate virtual environment and set LD_LIBRARY_PATH
echo "Activating virtual environment and setting LD_LIBRARY_PATH..."
VENV_PATH="$SCRATCH/new_vista_pytorch_venv" # Define path to your venv
if [ -d "$VENV_PATH" ]; then
  source "$VENV_PATH/bin/activate"
  VENV_LIB_PATH="$VENV_PATH/lib"  # Path to venv's lib directory
  export LD_LIBRARY_PATH="$VENV_LIB_PATH:$LD_LIBRARY_PATH" # Prepend venv lib path
  echo "Virtual environment activated from $VENV_PATH"
  echo "LD_LIBRARY_PATH updated to: $LD_LIBRARY_PATH"
else
  echo "Error: Virtual environment not found at $VENV_PATH"
  exit 1 # Exit if venv is not found
fi

# Print current directory and Python version
echo "Current directory: $(pwd)"
echo "Python version: $(python --version)"

# Check if data directory exists
DATA_DIR="data/navier_stokes/ns_data/16traj"
echo "Checking data directory: $DATA_DIR"
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found at $DATA_DIR"
    exit 1
fi

# List files in data directory
echo "Files in data directory:"
ls -l $DATA_DIR

# Run the Python script with unbuffered output
echo "Starting downsample_data.py..."
PYTHONUNBUFFERED=1 python -u downsample_data.py

# Check if the script completed successfully
if [ $? -eq 0 ]; then
    echo "downsample_data.py completed successfully"
else
    echo "Error: downsample_data.py failed"
    exit 1
fi
