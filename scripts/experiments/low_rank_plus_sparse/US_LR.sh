# Base command
base_cmd="torchrun --nproc_per_node=1"

composite_common_params=" \
    --opt.checkpointing False \
    --wandb.entity wandb_entity \
    --wandb.project tensorgrad \
    --opt.learning_rate 0.001 \
    --opt.n_epochs 500 \
    --opt.scheduler_T_max 500 \
    --data.batch_size 8 \
    --opt.update_proj_gap 1000 \
    --opt.enforce_full_complex_precision True \
    --opt.optimizer_type tensorgrad \
    --opt.second_proj_type low_rank \
    --opt.proj_type unstructured_sparse \
    --wandb.log False \
    --opt.tensorgrad True"



$base_cmd train_ns_repro_tensorgrad.py $composite_common_params \
    --wandb.name US_LR_25%_half \
    --fno.fno_block_precision mixed \
    --fno.fno_block_weights_precision half \
    --fno.stabilizer tanh \
    --opt.second_rank 0.20 \
    --opt.sparse_ratio 0.05 
    

$base_cmd train_ns_repro_tensorgrad.py $composite_common_params \
    --wandb.name US_LR_25%_full \
    --fno.fno_block_precision full \
    --fno.fno_block_weights_precision full \
    --opt.second_rank 0.20 \
    --opt.sparse_ratio 0.05 
