# Base command
base_cmd="torchrun --nproc_per_node=1"

# Common parameters for single unstructured sparse projection setup
composite_common_params=" \
    --opt.optimizer_type adamw \
    --opt.checkpointing False \
    --wandb.entity sloeschcke \
    --opt.n_epochs 500 \
    --opt.scheduler_T_max 500 \
    --data.batch_size 8 \
    --opt.learning_rate 0.001 \
    --opt.update_proj_gap 1000 \
    --opt.proj_type low_rank \
    --opt.second_proj_type unstructured_sparse \
    --opt.second_sparse_type randk \
    --opt.sparse_type randk \
    --opt.optimizer_type tensorgrad 
    --wandb.log False \
    --opt.tensorgrad True"


$base_cmd train_ns_repro_tensorgrad.py $composite_common_params \
    --wandb.name LR_US_25%_full \
    --fno.fno_block_precision full \
    --fno.fno_block_weights_precision full \
    --opt.second_sparse_ratio 0.05 \
    --opt.rank 0.20 

$base_cmd train_ns_repro_tensorgrad.py $composite_common_params \
    --wandb.name LR_US_25%_half \
    --fno.fno_block_precision mixed \
    --fno.fno_block_weights_precision half \
    --fno.stabilizer tanh \
    --opt.second_sparse_ratio 0.05 \
    --opt.rank 0.20 
