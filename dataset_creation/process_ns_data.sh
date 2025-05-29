# Create output directory if it doesn't exist
mkdir -p output_files

# Print hostname and show GPU info
echo "Running on $(hostname):"
nvidia-smi


# 16 TRAJ DATAset
INPUT_PATH="data/navier_stokes/ns_data/reassembled_file.pt"
TRAIN_PATH="data/navier_stokes/ns_data/16traj/1024_train_data_t4_T4.pt"
TEST_PATH="data/navier_stokes/ns_data/16traj/1024_test_data_t4_T4.pt"
SIDE_LENGTH=2048
MAX_SIDE_LENGTH=1024
T_USE=4
T=4

# Run the Python script
python process_ns_data.py \
    --input "$INPUT_PATH" \
    --train "$TRAIN_PATH" \
    --test "$TEST_PATH" \
    --t_use $T_USE \
    --T $T \
    --train_ratio 0.8 \
    --seed 42 \
    --side_length $SIDE_LENGTH \
    --max_side_length $MAX_SIDE_LENGTH \
    --verbose True \

echo "process_ns_data.py completed"

echo "start convert_pt_to_hdf5.py"
python convert_pt_to_hdf5.py
echo "convert_pt_to_hdf5.py completed"

echo "start inspect_data.py"
python inspect_data.py
echo "inspect_data.py completed"

echo "start downsampling"
python downsample_data.py
echo "downsample_data.py completed"



INPUT_PATH="data/navier_stokes/ns_data/re10000_grid=2048_2_N=1_dt=256.0_Ttj=10-40stat.pt"
TRAIN_PATH="data/navier_stokes/1024_new/1024_train_data.pt"
TEST_PATH="data/navier_stokes/1024_new/1024_test_data.pt"
SIDE_LENGTH=1024
MAX_SIDE_LENGTH=1024
T_USE=4
T=64


# Run the Python script
python process_ns_data.py \
    --input "$INPUT_PATH" \
    --train "$TRAIN_PATH" \
    --test "$TEST_PATH" \
    --t_use $T_USE \
    --T $T \
    --train_ratio 0.8 \
    --seed 42 \
    --side_length $SIDE_LENGTH \
    --max_side_length $MAX_SIDE_LENGTH \
    --verbose True \

echo "process_ns_data.py completed"

echo "start convert_pt_to_hdf5.py"
python convert_pt_to_hdf5.py
echo "convert_pt_to_hdf5.py completed"

echo "start inspect_data.py"
python inspect_data.py
echo "inspect_data.py completed"

echo "start downsampling"
python downsample_data.py
echo "downsample_data.py completed"
