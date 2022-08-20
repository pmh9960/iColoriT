export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=${1:-0}
MASTER_PORT=${2:-4885}

# path to imagenet-1k train set and validation dataset
DATA_PATH=''
VAL_DATA_PATH=''
VAL_HINT_DIR=''
# Set the path to save checkpoints
OUTPUT_DIR='checkpoints'
TB_LOG_DIR='tf_logs'

# other options
opt=${3:-}

    
# Training epochs used for pretrained iColoriT are
# Base  - 1000 epochs
# Small - 100 epochs
# Tiny  - 25 epochs
# all with a batch size of 256.
# Other hyper-parameters follow the default numbers. 


# batch_size can be adjusted according to the graphics card
python -m torch.distributed.launch --nproc_per_node=$(((${#CUDA_VISIBLE_DEVICES}+1)/2)) --master_port ${MASTER_PORT} \
    train.py \
    --data_path ${DATA_PATH} \
    --val_data_path ${VAL_DATA_PATH} \
    --val_hint_dir ${VAL_HINT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${TB_LOG_DIR} \
    --exp_name exp \
    --save_args_txt \
    $opt
