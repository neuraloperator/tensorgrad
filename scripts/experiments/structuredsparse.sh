# Base command
base_cmd="torchrun --nproc_per_node=1"

# Common parameters for single unstructured sparse projection setup
common_params=" \
    --opt.optimizer_type adamw \
    --opt.checkpointing False \
    --wandb.entity wandb_entity \
    --wandb.project tensorgrad \
    --opt.n_epochs 500 \
    --opt.scheduler_T_max 500 \
    --data.batch_size 8 \
    --opt.learning_rate 0.001 \
    --fno.fno_block_precision full \
    --fno.fno_block_weights_precision full \
    --wandb.log True \
    --opt.update_proj_gap 1000 \
    --opt.enforce_full_complex_precision True \
    --opt.optimizer_type adamw \
    --opt.tensorgrad True \
    --opt.scale_by_mask_ratio True \
    --opt.sparse_type randk \
    --opt.proj_type structured_sparse \
    --wandb.log False"
    #--opt.sparse_type randk \ # topk, probability, randk


$base_cmd train_ns_repro_tensorgrad.py $common_params \
    --wandb.name SS_25%_full \
    --fno.fno_block_precision full \
    --fno.fno_block_weights_precision full \
    --opt.sparse_ratio 0.25 

$base_cmd train_ns_repro_tensorgrad.py $common_params \
    --wandb.name SS_50%_full \
    --fno.fno_block_precision full \
    --fno.fno_block_weights_precision full \
    --opt.sparse_ratio 0.50 

$base_cmd train_ns_repro_tensorgrad.py $common_params \
    --wandb.name SS_25%_half \
    --fno.fno_block_precision mixed \
    --fno.fno_block_weights_precision half \
    --fno.stabilizer tanh \
    --opt.sparse_ratio 0.25 

$base_cmd train_ns_repro_tensorgrad.py $common_params \
    --wandb.name SS_50%_half \
    --fno.fno_block_precision mixed \
    --fno.fno_block_weights_precision half \
    --fno.stabilizer tanh \
    --opt.sparse_ratio 0.50 
