import os
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

# Set necessary environment variables
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"  # Choose an unused port
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"

dist.init_process_group(
    backend="gloo",
    init_method="env://",  # Use the updated init method
    rank=0,
    world_size=1
)

# Save checkpoint
ckpt_id = "ckpts/fallback_test_checkpoint"
checkpoint_data = {
    "app": {"model_state": "dummy_model_data", "optimizer_state": "dummy_optimizer_data"},
    "manifest": {"info": "test manifest"}
}
dcp.save(checkpoint_data, checkpoint_id=ckpt_id)

# Load checkpoint
loaded_data = {}
dcp.load(state_dict=loaded_data, checkpoint_id=ckpt_id)
print("Loaded checkpoint data:", loaded_data)

# make assertion and exception handling

if loaded_data == {}:
    print("#########################")
    print("loaded data with DCP is empty")
    print("#########################")
else:
    print('Success')
    
    
print("#########################")
print("Trying with torch.save")

import torch

torch.save(checkpoint_data, "ckpts/fallback_test_checkpoint.pt")
print("Checkpoint saved using torch.save!")

import torch
loaded_data = torch.load("ckpts/fallback_test_checkpoint.pt")
print("Loaded data using torch.load:", loaded_data)

if loaded_data == {}:
    print("#########################")
    print("loaded data with torch.save is empty")
    print("#########################")
else:
    print('Success with torch.save')