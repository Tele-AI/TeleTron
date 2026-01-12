export MASTER_ADDR=$GEMINI_HOST_IP_taskrole1_0
export RANK=$GEMINI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX
export NCCL_DEBUG=INFO&&export NCCL_ALGO=RING

cp -r /nvfile-heatstorage/model_zoo/Wan2___1-I2V-14B-480P/ /workspace/

cd /nvfile-heatstorage/yxy/code/Teletron/
pip install -r requirements.txt
bash examples/teleai/run_i2v.sh
