# !/bin/bash
# BSUB -q p1                          # Specify queue
# BSUB -J composite_proj              # Set the job name
# BSUB -n 16                          # Request number of cores (default: 1)
# BSUB -R "span[hosts=1]"             # Specify cores must be on the same host
# BSUB -R "rusage[mem=4GB]"          # Specify 32GB of memory per core/slot
# BSUB -W 10:00                       # Set walltime limit: hh:mm
# BSUB -o output_files/job.%J.out     # Specify the output file. %J is the job-id
# BSUB -e output_files/job.%J.err     # Specify the error file. %J is the job-id

# Requesting GPU resources
# BSUB -gpu "num=1:j_exclusive=yes"   # Request 1 GPU, with exclusive access

echo "Running on $(hostname):"
nvidia-smi

eval "$(conda shell.bash hook)"
conda activate tensorgalore

# Base command
base_cmd="torchrun --nproc_per_node=1"

# Common parameters for single unstructured sparse projection setup
common_params=" \
    --opt.optimizer_type adamw \
    --opt.checkpointing False \
    --wandb.entity sloeschcke \
    --wandb.project tensorgalore \
    --opt.n_epochs 500 \
    --opt.scheduler_T_max 500 \
    --data.batch_size 8 \
    --opt.learning_rate 0.001 \
    --fno.fno_block_precision full \
    --fno.fno_block_weights_precision full \
    --opt.update_proj_gap 1000 \
    --opt.tensorgrad True \
    --opt.enforce_full_complex_precision True \
    --opt.optimizer_type adamw \
    --opt.tensorgrad True \
    --opt.proj_type low_rank \
    --opt.rank 0.25 \
    --opt.naive_galore True \
    --wandb.log True"



# loop over first_dim_rollup
for first_dim_rollup in 0 1 2 3; do
    $base_cmd train_ns_repro_tensorgrad.py $common_params \
        --wandb.name LR_Naive_25%_full_roll_dim${first_dim_rollup} \
        --fno.fno_block_precision full \
        --fno.fno_block_weights_precision full \
        --opt.proj_type low_rank \
        --opt.first_dim_rollup ${first_dim_rollup}
done
