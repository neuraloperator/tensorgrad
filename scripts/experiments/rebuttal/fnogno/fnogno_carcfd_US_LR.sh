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


for lr in 0.0005 0.001  
do
    # Your LSF job parameters
    python train_fnogno_carcfd.py \
    --wandb.log True \
    --opt.n_epochs 100 \
    --opt.weight_decay 0.00025 \
    --opt.learning_rate $lr \
    --opt.tensorgrad True \
    --opt.optimizer_type tensorgrad \
    --opt.proj_type unstructured_sparse \
    --opt.sparse_ratio 0.05 \
    --opt.sparse_type randk \
    --opt.second_proj_type low_rank \
    --opt.second_rank 0.20 \
    --opt.adamw_support_complex True \
    --opt.enforce_full_complex_precision True \
    --fnogno.gno_radius 0.055 \
    --wandb.name US-LR_lr-${lr} \
    --wandb.group "paper-reproduction"
    #--fnogno.gno_radius 0.055 \
done

