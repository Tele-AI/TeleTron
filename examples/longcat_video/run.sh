#!/bin/bash

# Run model
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FUSED_ATTN=0
export NVTE_FLASH_ATTN=1
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/ai_infra/code/lit117/Megatron-LM
export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/ai_infra/code/lit117/yuc/env/teleai_data_tool
export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/ai_infra/code/lit117/qiuyang/Video-Depth-Anything/
export PYTHONPATH=$PYTHONPATH:$(pwd)

####################################### IMPORTANT ARGS #######################################
# Debug params for 1-layer model
export DEBUG_DEPTH=1
export DEBUG_HIDDEN_SIZE=128
export DEBUG_NUM_HEADS=4

# Parallel config 
CP=1
TP=1 # not support

# Multi-node config 
N_MOE=1
N_GPU_FOR_TRAIN=1
N_GPU_FOR_DATA=1 # Disable distributed VAE GPUs

# Single-node config 
EXPR_NAME=longcat_debug

TRAIN_SCRIPT=${1:-"examples/longcat_video/pretrain_t2v.py"}
CONFIG_PATH=${2:-"examples.longcat_video.config.longcat_t2v.config"}
shift
echo "Launching: $TRAIN_SCRIPT"

TENSORBOARD_LOGS_PATH=./logs/${EXPR_NAME}
CHECKPOINT_PATH_LOAD=/nvfile-heatstorage/myk/Teletron/checkpoint/${EXPR_NAME}
CHECKPOINT_PATH_SAVE=/nvfile-heatstorage/myk/Teletron/checkpoint/${EXPR_NAME}
####################################### IMPORTANT ARGS END #######################################

mkdir -p $CHECKPOINT_PATH_SAVE

MASTER_ADDR=${MASTER_ADDR:-'127.0.0.1'}
MASTER_PORT='11392'
NODE_RANK=${RANK:-'0'}

MBS=1
N_GPU=$((N_GPU_FOR_TRAIN+N_GPU_FOR_DATA))
NNODES=$((($N_GPU-1)/8+1))
WORLD_SIZE=$N_GPU_FOR_TRAIN

N_VAE=$N_GPU_FOR_DATA
GBS=$(($WORLD_SIZE*$MBS/$CP/$TP))

if [ $NNODES -eq 1 ]; then
    N_PROC=$N_GPU
else
    N_PROC=8
fi

echo '$MASTER_ADDR' $MASTER_ADDR
echo '$NODE_RANK & $NNODES' $NODE_RANK $NNODES
echo '$N_GPU_FOR_TRAIN' $N_GPU_FOR_TRAIN
echo '$N_GPU_FOR_DATA' $N_GPU_FOR_DATA

DISTRIBUTED_ARGS=(
    --nproc_per_node $N_PROC 
    --nnodes $NNODES 
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)


TRAINING_ARGS=( 
    --micro-batch-size ${MBS}
    --train-iters 10
    --weight-decay 1e-4
    --init-method-std 0.006 
    --clip-grad 1.0
    --bf16
    --lr 1e-5
    --lr-decay-style constant
    --lr-warmup-fraction 0
    --recompute-granularity full 
    --recompute-method block 
    # --activation-offload
    --use-distributed-optimizer
    --recompute-num-layers 1
    --no-rope-fusion
    --distributed-timeout-minutes 60
    --override-opt_param-scheduler
    --data-parallel-random-init
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP}
    --context-parallel-size ${CP}
    --distributed-vae # Disabled for single GPU debug
    --distributed-vae-world-size  1
    --consumer-models-num $N_MOE
)
DATA_ARGS=(
    --split 949,50,1
    --num-workers 0 # Set to 0 for debug
    --config-path ${CONFIG_PATH}
)

EVAL_AND_LOGGING_ARGS=(
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    --tensorboard-log-interval 1
    --tensorboard-queue-size 10
    --log-interval 1 # for terminal infos
    --save-interval 500
    --eval-interval 500
    # --load $CHECKPOINT_PATH_LOAD 
    # --save $CHECKPOINT_PATH_SAVE
    --eval-iters 1 # sample 1 video to eval
    --producer-log-level 1 # 1: debug | 2: Info
)

torchrun ${DISTRIBUTED_ARGS[@]} ${TRAIN_SCRIPT} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]}    \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${LORA_CFG[@]} \
    "$@"
