# Create output directories if they don't exist
mkdir -p profiler_outputs

# Common parameters
common_params=" \
    --opt.checkpointing False \
    --wandb.entity sloeschcke \
    --opt.tensorgrad False \
    --opt.n_epochs 1 \
    --opt.scheduler_T_max 1 \
    --fno.n_modes [128,128] \
    --fno.hidden_channels 256 \
    --data.train_resolution 1024 \
    --data.batch_size 1 \
    --opt.resume_from_dir None \
    --opt.profiling True \
    --opt.save_dir "./profiler_outputs"
    --opt.save_every 100 \
    --fno.activation_checkpoint False \
    --opt.optimizer_type adamw \
    --opt.tensorgrad False \
    --wandb.log False"

# Half precision run
python train_ns_repro_tensorgrad.py $common_params \
    --fno.fno_block_precision mixed \
    --fno.fno_block_weights_precision half \
    --wandb.name adamw_half

# Full precision run
python train_ns_repro_tensorgrad.py $common_params \
    --fno.fno_block_precision full \
    --fno.fno_block_weights_precision full \
    --wandb.name adamw_full

