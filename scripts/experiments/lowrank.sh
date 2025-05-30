# Base command
base_cmd="torchrun --nproc_per_node=1"

# Common parameters for single unstructured sparse projection setup
common_params=" \
    --opt.optimizer_type adamw \
    --opt.checkpointing False \
    --wandb.entity wandb_entity \
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
    --wandb.log False"



$base_cmd train_ns_repro_tensorgrad.py $common_params \
    --wandb.name LR_25%_full \
    --fno.fno_block_precision full \
    --fno.fno_block_weights_precision full \
    --opt.proj_type low_rank \
    


$base_cmd train_ns_repro_tensorgrad.py $common_params \
    --wandb.name LR_25%_half \
    --fno.fno_block_precision mixed \
    --fno.fno_block_weights_precision half \
    --fno.stabilizer tanh
