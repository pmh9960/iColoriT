export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=${1:-0}

PRED_DIR=''
GT_DIR=''
NUM_HINT=${2:-10}

# other options
opt=${3:-}

# batch_size can be adjusted according to the graphics card
python evaluation/evaluate.py \
    --pred_dir=${PRED_DIR} \
    --gt_dir=${GT_DIR} \
    --num_hint=${NUM_HINT} \
    $opt