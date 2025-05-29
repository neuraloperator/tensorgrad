# Base command
base_cmd="torchrun --nproc_per_node=1"

composite_common_params=" \
    --opt.checkpointing False \
    --wandb.entity wandb_entity \
    --opt.learning_rate 0.001 \
    --opt.n_epochs 500 \
    --opt.scheduler_T_max 500 \
    --data.batch_size 8 \
    --opt.update_proj_gap 1000 \
    --opt.enforce_full_complex_precision True \
    --opt.optimizer_type tensorgrad \
    --wandb.log False \
    --opt.second_proj_type low_rank \
    --opt.proj_type structured_sparse \
    --opt.second_rank 0.20 \
    --opt.sparse_ratio 0.05 \
    --opt.tensorgrad True"

$base_cmd train_ns_repro_tensorgrad.py $composite_common_params \
    --wandb.name SS_LR_25%_full \
    --fno.fno_block_precision full \
    --fno.fno_block_weights_precision full 
    
        
    
$base_cmd train_ns_repro_tensorgrad.py $composite_common_params \
    --wandb.name SS_LR_25%_half \
    --fno.fno_block_precision mixed \
    --fno.fno_block_weights_precision half \
    --fno.stabilizer tanh \
