#!/bin/bash
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FUSED_ATTN=0
export NVTE_FLASH_ATTN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Debug NCCL
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # Force TCP first to debug connection
export NCCL_P2P_DISABLE=1 # Disable P2P to rule out PCIe issues
export NCCL_SOCKET_IFNAME=^lo,docker,veth  # Exclude loopback and docker interfaces

export TELETRON_ZERO2_REDUCE_BUCKET_SIZE=50000000
export TELETRON_ZERO2_ALLGATHER_BUCKET_SIZE=50000000
export TELETRON_ZERO2_CONTIGUOUS_GRADIENTS=0
export LONGCAT_DST_SIZE=${LONGCAT_DST_SIZE:-"256,256"}

# Rank config from argument $1 (default 0)
RANK_ARG=${1:-0}
if [ "$RANK_ARG" -eq 0 ]; then
    echo "Running as Master Node (Rank 0, 2 GPUs)"
    export NODE_RANK=0
    export CUDA_VISIBLE_DEVICES=2,3
elif [ "$RANK_ARG" -eq 1 ]; then
    echo "Running as Worker Node (Rank 1, 8 GPUs)"
    export NODE_RANK=1
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
else
    echo "Unknown Rank argument: $RANK_ARG. Using env defaults."
    export NODE_RANK=${RANK:-0}
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
fi

export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/ai_infra/code/lit117/Megatron-LM
export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/ai_infra/code/lit117/yuc/env/teleai_data_tool
export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/ai_infra/code/lit117/qiuyang/Video-Depth-Anything/
export PYTHONPATH=$PYTHONPATH:$(pwd)

####################################### IMPORTANT ARGS #######################################
# Debug params for 1-layer model
export DEBUG_DEPTH=40
export DEBUG_HIDDEN_SIZE=5120
export DEBUG_NUM_HEADS=40

# Parallel config 
CP=4
TP=1 # not support

# Multi-node config 
N_MOE=1
N_GPU_FOR_TRAIN=8
N_GPU_FOR_DATA=2

# Single-node config 
EXPR_NAME=longcat_i2v_debug
N_VAE=$N_GPU_FOR_DATA
TRAIN_SCRIPT=${2:-"examples/longcat_video/pretrain_i2v.py"}
CONFIG_PATH=${3:-"examples.longcat_video.config.longcat_i2v.config"}
if [ $# -ge 3 ]; then
    shift 3
elif [ $# -ge 2 ]; then
    shift 2
elif [ $# -ge 1 ]; then
    shift 1
fi
echo "Launching: $TRAIN_SCRIPT"

TENSORBOARD_LOGS_PATH=./logs/${EXPR_NAME}
CHECKPOINT_PATH_LOAD=/nvfile-heatstorage/myk/Teletron/checkpoint/${EXPR_NAME}
CHECKPOINT_PATH_SAVE=/nvfile-heatstorage/myk/Teletron/checkpoint/${EXPR_NAME}
####################################### IMPORTANT ARGS END #######################################

mkdir -p $CHECKPOINT_PATH_SAVE

MASTER_ADDR=${MASTER_ADDR:-'10.127.16.32'}
MASTER_PORT=${MASTER_PORT:-'8088'}
NNODES=2
NODE_RANK=${NODE_RANK:-${RANK:-'0'}}

MBS=1
N_GPU=$((N_GPU_FOR_TRAIN+N_GPU_FOR_DATA))
WORLD_SIZE=$N_GPU_FOR_TRAIN
GPUS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | awk -F"," '{print NF}')

N_VAE=$N_GPU_FOR_DATA
GBS=$(($WORLD_SIZE*$MBS/$CP/$TP))

if [ $NNODES -eq 1 ] && [ $N_GPU -ne $GPUS_PER_NODE ]; then
    echo "Invalid GPU config: CUDA_VISIBLE_DEVICES has $GPUS_PER_NODE GPUs but N_GPU_FOR_TRAIN+N_GPU_FOR_DATA=$N_GPU"
    exit 1
fi

if [ -z "${LONGCAT_CP_SPLIT_HW}" ]; then
    if [ $CP -eq 6 ]; then
        export LONGCAT_CP_SPLIT_HW="3,2"
    elif [ $CP -eq 4 ]; then
        export LONGCAT_CP_SPLIT_HW="2,2"
    elif [ $CP -eq 3 ]; then
        export LONGCAT_CP_SPLIT_HW="3,1"
    elif [ $CP -eq 2 ]; then
        export LONGCAT_CP_SPLIT_HW="2,1"
    elif [ $CP -eq 1 ]; then
        export LONGCAT_CP_SPLIT_HW="1,1"
    else
        export LONGCAT_CP_SPLIT_HW="${CP},1"
    fi
fi

if [ $((N_GPU_FOR_TRAIN % (CP * TP))) -ne 0 ]; then
    echo "Invalid parallel config: N_GPU_FOR_TRAIN=$N_GPU_FOR_TRAIN must be divisible by CP*TP=$((CP * TP))"
    exit 1
fi

if [ $N_GPU_FOR_DATA -gt 0 ] && [ $(((N_GPU_FOR_TRAIN / (CP * TP)) % N_GPU_FOR_DATA)) -ne 0 ]; then
    echo "Invalid VAE/CP config: (N_GPU_FOR_TRAIN/(CP*TP))=$((N_GPU_FOR_TRAIN / (CP * TP))) must be divisible by N_GPU_FOR_DATA=$N_GPU_FOR_DATA"
    exit 1
fi

N_PROC=$GPUS_PER_NODE

echo '$MASTER_ADDR' $MASTER_ADDR
echo '$NODE_RANK & $NNODES' $NODE_RANK $NNODES
echo '$N_GPU_FOR_TRAIN' $N_GPU_FOR_TRAIN
echo '$N_GPU_FOR_DATA' $N_GPU_FOR_DATA
echo '$GPUS_PER_NODE' $GPUS_PER_NODE

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
    --clip-grad 0.0
    --fp16
    --lr 1e-5
    --lr-decay-style constant
    --lr-warmup-fraction 0
    --recompute-granularity full 
    --recompute-method block 
    --activation-offload
    --use-distributed-optimizer
    --recompute-num-layers 100
    --use-zero2
    --no-rope-fusion
    --distributed-timeout-minutes 60
    --override-opt_param-scheduler
    --data-parallel-random-init
    --empty-unused-memory-level 2
    --manual-gc
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP}
    --context-parallel-size ${CP}
    --distributed-vae # Disabled for single GPU debug
    --distributed-vae-world-size $N_VAE
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
