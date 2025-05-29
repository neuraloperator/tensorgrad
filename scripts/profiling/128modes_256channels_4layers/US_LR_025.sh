# Create output directories if they don't exist
mkdir -p profiler_outputs

# Common parameters
common_params=" \
    --opt.checkpointing False \
    --opt.tensorgrad True \
    --opt.n_epochs 1 \
    --opt.scheduler_T_max 1 \
    --opt.update_proj_gap 100 \
    --fno.n_modes [128,128] \
    --fno.hidden_channels 256 \
    --data.train_resolution 1024 \
    --data.batch_size 1 \
    --opt.resume_from_dir None \
    --opt.profiling True \
    --opt.save_dir "./profiler_outputs"
    --opt.save_every 100 \
    --opt.per_layer_opt False \
    --fno.activation_checkpoint True \
    --opt.n_iter_max_tucker 1 \
    --opt.optimizer_type tensorgrad \
    --opt.second_proj_type low_rank \
    --opt.second_rank 0.20 \
    --opt.svd_type randomized_svd \
    --opt.proj_type unstructured_sparse \
    --opt.sparse_type randk \
    --opt.sparse_ratio 0.05 \
    --opt.enforce_full_complex_precision True \
    --opt.tucker_warm_restart True \
    --wandb.log False"

# Half precision run
python train_ns_repro_tensorgrad.py $common_params \
    --fno.fno_block_precision mixed \
    --fno.fno_block_weights_precision half \
    --wandb.name US_LR_025_half

# Full precision run
python train_ns_repro_tensorgrad.py $common_params \
    --fno.fno_block_precision full \
    --fno.fno_block_weights_precision full \
    --wandb.name US_LR_025_full  