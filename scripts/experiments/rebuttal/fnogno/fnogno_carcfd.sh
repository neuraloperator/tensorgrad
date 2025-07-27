# !/bin/bash
# BSUB -q p1                          # Specify queue
# BSUB -J fnogno_carcfd               # Set the job name
# BSUB -n 16                          # Request number of cores (default: 1)
# BSUB -R "span[hosts=1]"             # Specify cores must be on the same host
# BSUB -R "rusage[mem=32GB]"          # Specify 32GB of memory per core/slot
# BSUB -W 72:00                       # Set walltime limit: hh:mm
# BSUB -o output_files/job.%J.out     # Specify the output file. %J is the job-id
# BSUB -e output_files/job.%J.err     # Specify the error file. %J is the job-id

# Requesting GPU resources
# BSUB -gpu "num=1:j_exclusive=yes"   # Request 1 GPU, with exclusive access

echo "Running on $(hostname):"
nvidia-smi

eval "$(conda shell.bash hook)"
conda activate tensorgrad

# Base command
base_cmd="python"

common_params=" \
    --config_file fnogno_carcfd_config.yaml \
    --wandb.entity sloeschcke \
    --wandb.project fnogno_carcfd \
    --data.n_train 5 \
    --data.n_test 5 \
    --wandb.log False"

# Experiment 1: Shape-Net Car reproduction (target ~7.12% test error)
$base_cmd train_fnogno_carcfd.py $common_params \
    --wandb.name fnogno_shapenet_car_paper \
    --wandb.group "paper-reproduction"

