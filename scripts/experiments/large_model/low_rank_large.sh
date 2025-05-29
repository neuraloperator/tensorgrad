echo "Running on $(hostname):"
nvidia-smi

eval "$(conda shell.bash hook)"
conda activate tensorgrad

# ==============================================================================
# Configuration
# ==============================================================================
config_file="ns_tensorgrad_repro_config_1024_16.yaml"

# ==============================================================================
# Common Arguments
# ==============================================================================
common_args="--config_file ${config_file} \
    --data.train_resolution 256 \
    --data.test_resolutions [512,1024] \
    --data.test_batch_sizes [16,8] \
    --data.n_tests [100,100] \
    --data.download False \
    --data.batch_size 8 \
    --opt.checkpointing True \
    --wandb.entity wandb-entity \
    --opt.tensorgrad True \
    --verbose True \
    --opt.training_loss l2 \
    --data.root data/navier_stokes/ns_data/processed_traj1_T256_t4_dataset \
    --fno.n_modes [128,128] \
    --fno.hidden_channels 128 \
    --fno.n_layers 4 \
    --fno.fno_block_precision full \
    --fno.fno_block_weights_precision full \
    --opt.update_proj_gap_mode fixed \
    --opt.update_proj_gap 500 \
    --opt.proj_type low_rank \
    --opt.rank 0.20 \
    --fno.projection_channel_ratio 4 \
    --opt.save_every 1 \
    --opt.step_size 20 \
    --opt.scheduler_T_max 100 \
    --opt.optimizer_type adamw \
    --opt.resume_from_dir ckpts \
    --wandb.log False"

# ==============================================================================
# Training on 256x256 Resolution
# ==============================================================================
echo "Starting training on 256x256 resolution..."
torchrun --nproc_per_node=1 train_ns_repro_tensorgrad.py \
    ${common_args} \
    --opt.learning_rate 0.005 \
    --data.train_resolution 256 \
    --data.test_resolutions [1024] \
    --data.test_batch_sizes [8] \
    --data.n_tests [100] \
    --opt.n_epochs 50 \
    --wandb.name low_rank_25

# ==============================================================================
# Fine-tuning on 1024x1024 Resolution
# ==============================================================================
echo "Starting fine-tuning on 1024x1024 resolution..."
torchrun --nproc_per_node=1 train_ns_repro_tensorgrad.py \
    ${common_args} \
    --opt.learning_rate 0.005 \
    --data.train_resolution 1024 \
    --data.test_resolutions [1024] \
    --data.test_batch_sizes [8] \
    --data.n_tests [100] \
    --opt.n_epochs 60 \
    --data.batch_size 2 \
    --wandb.name low_rank_25

echo "TensorGRaD large model training complete!"