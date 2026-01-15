#!/bin/bash

# Run model
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FUSED_ATTN=0
export NVTE_FLASH_ATTN=1
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


# TODO, change to your own path
# export PYTHONPATH=
export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/ai_infra/code/lit117/Megatron-LM
# export MEMORY_SNAPSHOT=True
# export PROF_SAVE_PATH="./log_memory_0607_2"

####################################### 
# TODO: set config below
# TODO: Recommended ratio: N_GPU_FOR_TRAIN / N_MOE / CP <= N_GPU_FOR_DATA 
# TODO: Constrain: N_GPU_FOR_TRAIN = N_MOE * CP * N

# Parallel config 
CP=2
TP=1 # not support

# Multi-node config 
N_MOE=1
N_LAYERS=1
N_GPU_FOR_TRAIN=4
N_GPU_FOR_DATA=2

# Single-node config 
# N_MOE=1
# N_LAYERS=1
# N_GPU_FOR_TRAIN=1
# N_GPU_FOR_DATA=1

TENSORBOARD_LOGS_PATH=./logs
# CHECKPOINT_PATH_LOAD=/nvfile-heatstorage/yxy/code/Teletron/debug/ckpt/wan_layer25_i2v/refactor/ckpt/teletron
# CHECKPOINT_PATH_SAVE=/nvfile-heatstorage/yxy/code/Teletron/debug/ckpt/wan_layer25_i2v/refactor/expr4_transform
CHECKPOINT_PATH=/nvfile-heatstorage/yuc/refactor/Teletron/test
mkdir -p $CHECKPOINT_PATH

####################################### 

MASTER_ADDR=${MASTER_ADDR:-'127.0.0.1'}
MASTER_PORT='11322'
NODE_RANK=${RANK:-'0'}

MBS=1
N_GPU=$((N_GPU_FOR_TRAIN+N_GPU_FOR_DATA))
NNODES=$((($N_GPU-1)/8+1))
WORLD_SIZE=$N_GPU_FOR_TRAIN
N_VAE=$N_GPU_FOR_DATA
GBS=$(($WORLD_SIZE*$MBS/$CP/$TP))
TOTAL_MOE_NODES=$((NNODES - N_VAE / 8))
NODES_PER_MOE=$((TOTAL_MOE_NODES / N_MOE))
I_MOE=$((NODE_RANK / NODES_PER_MOE))
if [ $NNODES -eq 1 ]; then
    N_PROC=$N_GPU
else
    N_PROC=8
fi
if [ $N_MOE -eq 1 ]; then
    MOE_ARGS=(
        --moe-step-factor-list 0.0 
        --moe-step-factor-list 1.0 
    )
elif [ $N_MOE -eq 2 ]; then
    MOE_ARGS=(
        --moe-step-factor-list 0.0 
        --moe-step-factor-list 0.833 
        --moe-step-factor-list 1.0
    )
elif [ $N_MOE -eq 4 ]; then
    MOE_ARGS=(
        --moe-step-factor-list 0.0 
        --moe-step-factor-list 0.625 
        --moe-step-factor-list 0.833
        --moe-step-factor-list 0.937 
        --moe-step-factor-list 1.0
    )
else
    echo "N_MOE must be 1, 2 or 4"
    exit 1
fi

echo '$MASTER_ADDR' $MASTER_ADDR
echo '$I_MOE & $N_MOE' $I_MOE $N_MOE
echo '$NODE_RANK & $NNODES' $NODE_RANK $NNODES
echo '$N_GPU_FOR_TRAIN' $N_GPU_FOR_TRAIN
echo '$N_GPU_FOR_DATA' $N_GPU_FOR_DATA


MERGE_FILE=/nvfile-heatstorage/teleai-infra/wxe/Megatron-LM/data/gpt_2_merge.txt
DATA_PATH=./checkpoint


DISTRIBUTED_ARGS=(
    --nproc_per_node $N_PROC 
    --nnodes $NNODES 
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --num-layers $N_LAYERS
    --hidden-size 5120        
    --num-attention-heads 40
    --has-image-input
    --seq-length 512          
    --max-position-embeddings 4096
    --tokenizer-type NullTokenizer
    --vocab-size 0
)

TRAINING_ARGS=(
    --micro-batch-size ${MBS}
    --train-iters 100000
    --weight-decay 1e-3
    --init-method-std 0.006 
    --clip-grad 0.0
    --bf16
    --lr 1e-5
    --lr-decay-style cosine
    --lr-warmup-fraction 0
    --recompute-granularity full 
    --recompute-method block 
    --activation-offload
    --use-distributed-optimizer
    --recompute-num-layers 40
    --no-rope-fusion
    --distributed-timeout-minutes 60
    # --distribute-saved-activations
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP}
    --context-parallel-size ${CP}
    --distributed-vae
    --distributed-vae-world-size $N_VAE
    --consumer-models-num $N_MOE
    --producer-batch-size 1
)
DATA_ARGS=(
    --dataset-type VastDataset
    --data-path $DATA_PATH 
    --merge-file $MERGE_FILE 
    --split 949,50,1
    --dataloader-type single
    --num-workers 1
)

EVAL_AND_LOGGING_ARGS=(
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    --tensorboard-log-interval 1
    --tensorboard-queue-size 10
    --log-interval 1 # for terminal infos
    --save-interval 2000
    --eval-interval 2000
    # --load $CHECKPOINT_PATH_LOAD 
    # --save $CHECKPOINT_PATH_SAVE/node_$I_MOE
    # --load $CHECKPOINT_PATH/node_$I_MOE
    --save $CHECKPOINT_PATH/node_$I_MOE
    --eval-iters 2 # sample 20 video to eval
)


# When using lora and resume train from breakpoint, need to provide a base model path
# as lora only save the adpater weight.
# Set the LOAD args in EVAL_AND_LOGGING_ARGS to your saved ckpt path.
# Training from start is no need to provide LORA_BASE_MODEL_PATH and LOAD args.
# LORA_CFG=(
#     --lora False 
#     --lora-rank 8
#     --lora-alpha 32
#     --lora-dropout 0.05
#     --lora-target-modules q,v # usage: q,v,k,o or q
#     --lora-bias none
#     --lora-task-type FEATURE_EXTRACTION
#     --lora-base-model-path # specify one if using lora
# )

# export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_IBEXT_DISABLE=1
# echo $NCCL_SOCKET_IFNAME
# echo $NCCL_IB_DISABLE
# echo $NCCL_IBEXT_DISABLE
# export NCCL_DEBUG=INFO

torchrun ${DISTRIBUTED_ARGS[@]} examples/teleai/pretrain_i2v.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]}    \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${LORA_CFG[@]} \
    "$@"
