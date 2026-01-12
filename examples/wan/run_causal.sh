export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FUSED_ATTN=0
export NVTE_FLASH_ATTN=1
export CUDA_VISIBLE_DEVICES=4,5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export GPUS_PER_NODE=2
export MASTER_ADDR=$GEMINI_HOST_IP_taskrole1_0
export MASTER_PORT=1234

export MASTER_ADDR=${MASTER_ADDR:-'127.0.0.1'}
export MASTER_PORT=${MASTER_PORT:-'12367'}
export NNODES=1
export NODE_RANK=0
# export WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
# export WORLD_SIZE=1

export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/ai_infra/code/lit117/Megatron-LM
export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/ai_infra/code/lit117/yuc/env/teleai_data_tool
export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/ai_infra/code/lit117/qiuyang/Video-Depth-Anything
CHECKPOINT_PATH_LOAD=None
CHECKPOINT_PATH_SAVE=/nvfile-heatstorage/teleai-infra/kaikai/examples
# mkdir -p $CHECKPOINT_PATH_SAVE
CONFIG_PATH=config.prone10_lowerlr.config #config.wan_autoregressive.config

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
)  

TRAINING_ARGS=(
    --task-type wan_autoregressive
    --lr 1e-5
    --train-iters 2000
    --weight-decay 0.0
    --adam-beta1 0.0
    --adam-beta2 0.999
    --use-distributed-optimizer
    --bf16 
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --consumer-models-num 1
    --distributed-vae
    --distributed-vae-world-size 1
# 
)

DATA_ARGS=(
    --micro-batch-size 1
    --config-path ${CONFIG_PATH}
)


EVAL_AND_LOGGING_ARGS=(
    --save $CHECKPOINT_PATH_SAVE
    --load $CHECKPOINT_PATH_LOAD 
    --save-interval 250
)


torchrun ${DISTRIBUTED_ARGS[@]} examples/wan/pretrain_causalwan.py \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]}    \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${LORA_CFG[@]} \
    "$@"
