#!/bin/bash

# Runs the "175B" parameter model
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FUSED_ATTN=0
export NVTE_FLASH_ATTN=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ENCODER_MODEL_PATH=/workspace/Wan2___1-I2V-14B-480P
ENCODER_TOKENIZER_PATH=/workspace/Wan2___1-I2V-14B-480P/google/umt5-xxl
MERGE_FILE=/nvfile-heatstorage/teleai-infra/wxe/Megatron-LM/data/gpt_2_merge.txt
DATA_PATH=./checkpoint
CONFIG_PATH=config.prone10_lowerlr.config

TP=1
CP=2
MBS=1

N_MOE=1
N_LAYERS=20
N_GPU_FOR_TRAIN=4
N_GPU_FOR_DATA=1

# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-'127.0.0.1'}
MASTER_PORT='11220'
NNODES=${WORLD_SIZE:-'1'}
NODE_RANK=${RANK:-'0'}


N_GPU=$((N_GPU_FOR_TRAIN+N_GPU_FOR_DATA))
WORLD_SIZE=$N_GPU_FOR_TRAIN
GBS=$(($WORLD_SIZE*$MBS/$CP/$TP))
GPUS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | awk -F"," '{print NF}')

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

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NNODES 
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --num-layers $N_LAYERS
    --hidden-size 5120        
    --num-attention-heads 40
    --seq-length 512          
    --max-position-embeddings 4096
    --tokenizer-type NullTokenizer
    --vocab-size 0
)

TRAINING_ARGS=(
    --model ParallelWanModel # TODO support more models
    --task-type wan_i2v_prone
    --micro-batch-size ${MBS}
    --train-iters 10000
    --weight-decay 1e-2
    --init-method-std 0.006 
    --clip-grad 0.0
    --bf16
    --lr 1e-5
    --lr-decay-style constant
    --lr-warmup-fraction 0
    --recompute-granularity full 
    --recompute-method block 
    --activation-offload
    --use-zero2
    --recompute-num-layers 40
    --no-rope-fusion
    --distributed-timeout-minutes 60
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP}
    --context-parallel-size ${CP}
    --distributed-vae
    --distributed-vae-world-size $N_GPU_FOR_DATA
    --consumer-models-num $N_MOE
)
DATA_ARGS=(
    --dataset-type VastDataset
    --data-path $DATA_PATH 
    --merge-file $MERGE_FILE 
    --split 949,50,1
    --dataloader-type single
    --num-workers 1
    --config-path ${CONFIG_PATH}
)

EVAL_AND_LOGGING_ARGS=(
    --encoder-model-path ${ENCODER_MODEL_PATH}
    --encoder-tokenizer-path ${ENCODER_TOKENIZER_PATH}
    --tensorboard-queue-size 10
    --log-interval 1
    --save-interval 10000
    --eval-interval 10000 
    --eval-iters 10000
)


torchrun ${DISTRIBUTED_ARGS[@]} examples/wan/pretrain_wan2_2.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]}    \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${LORA_CFG[@]} \
    "$@" 2>&1 | tee wan2_2.log
