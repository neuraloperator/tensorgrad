# !/bin/bash
# BSUB -q p1                          # Specify queue
# BSUB -J composite_proj              # Set the job name
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
base_cmd="torchrun --nproc_per_node=1"


composite_common_params=" \
    --opt.checkpointing False \
    --wandb.entity sloeschcke \
    --wandb.project tensorgalore \
    --opt.learning_rate 0.001 \
    --opt.n_epochs 500 \
    --opt.scheduler_T_max 500 \
    --data.batch_size 8 \
    --opt.update_proj_gap 1000 \
    --opt.enforce_full_complex_precision True \
    --opt.optimizer_type tensorgrad \
    --wandb.log True \
    --opt.tensorgrad False"

for seed in 0 1; do
    $base_cmd train_ns_repro_tensorgrad.py $composite_common_params \
        --wandb.name adamw_full__seed_$seed \
        --fno.fno_block_precision full \
        --fno.fno_block_weights_precision full \
        --distributed.seed $seed



    $base_cmd train_ns_repro_tensorgrad.py $composite_common_params \
        --wandb.name adamw_half__seed_$seed \
        --fno.fno_block_precision mixed \
        --fno.fno_block_weights_precision half \
        --distributed.seed $seed
done