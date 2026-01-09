#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

export NVTE_FUSED_ATTN="${NVTE_FUSED_ATTN:-0}"
export NVTE_FLASH_ATTN="${NVTE_FLASH_ATTN:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

add_path_if_exists() {
  local p="$1"
  if [ -d "$p" ]; then
    export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$p"
  fi
}

add_path_if_exists "/nvfile-heatstorage/ai_infra/code/lit117/Megatron-LM"
add_path_if_exists "/nvfile-heatstorage/ai_infra/code/lit117/yuc/env/teleai_data_tool"
add_path_if_exists "/nvfile-heatstorage/ai_infra/code/lit117/qiuyang/Video-Depth-Anything"

[ -n "${MEGATRON_PATH:-}" ] && add_path_if_exists "$MEGATRON_PATH"
[ -n "${TELEAI_DATA_TOOL_PATH:-}" ] && add_path_if_exists "$TELEAI_DATA_TOOL_PATH"
[ -n "${VIDEO_DEPTH_ANYTHING_PATH:-}" ] && add_path_if_exists "$VIDEO_DEPTH_ANYTHING_PATH"
[ -n "${EXTRA_PY_PATH:-}" ] && add_path_if_exists "$EXTRA_PY_PATH"

add_path_if_exists "$PROJECT_ROOT"
add_path_if_exists "$PROJECT_ROOT/tests"

export TELETRON_OPTIM_DIR="${TELETRON_OPTIM_DIR:-/nvfile-heatstorage/AIGC_H100/congliu/checkpoint/streaming_continue_1000_step/iter_0001000/mp_rank_00}"
export DESIRED_DP_COUNT="${DESIRED_DP_COUNT:-64}"
export ZERO2_SUBPROC_PER_RANK="${ZERO2_SUBPROC_PER_RANK:-16}"

TEST_FILE="${SCRIPT_DIR}/test_zero2_real_load.py"


TEST_TARGET="${TEST_TARGET:-${TEST_FILE}::TestDeepSpeedZero2LoadCheckpoint::test_deepspeed_zero2_load_checkpoint_end_to_end}"
python -m pytest -s -q "${TEST_TARGET}"
