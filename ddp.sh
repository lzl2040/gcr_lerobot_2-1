#!/bin/bash
# export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
# export NCCL_IB_DISABLE=0
# export NCCL_SOCKET_IFNAME=en,eth,em,bond,ib #bond0
# export NCCL_DEBUG=INFO
# export NCCL_NVLS_ENABLE=0

# export CFLAGS="-I/usr/include"
# export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"
# export CUTLASS_PATH="/path/to/cutlass"

deepspeed lerobot/scripts/dps_train.py \
    --deepspeed="./ds_zero2.json" \
    --policy.type="qwen" \
    --dataset.repo_id="whatever" \
    --dataset.processor="/datassd_1T/qwen25vl/Qwen2.5-VL-7B-Instruct/" \
    --dataset.parent_dir="/data_16T/lerobot_openx/" \
    --output_dir="/data_16T/deepseek/qwen_flow/"