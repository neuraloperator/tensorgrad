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


#for lr in 0.0005 0.001
#for lr in 0.00025 0.0005 0.001
for lr in 0.00025 
do
    # Your LSF job parameters
    python train_fnogno_carcfd.py \
        --opt.n_epochs 100 \
        --opt.step_size 100 \
        --opt.weight_decay 0.001 \
        --opt.learning_rate $lr \
        --fnogno.gno_radius 0.055 \
        --fnogno.fno_n_modes "[32,32,32]" \
        --wandb.name 32mode_lr-${lr} \
        --fnogno.gno_coord_embed_dim 32 \
        --wandb.group "paper-reproduction"
done

#fno_n_modes: List[int] = [16, 16, 16]