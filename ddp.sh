#!/bin/bash
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=en,eth,em,bond,ib #bond0
export NCCL_DEBUG=INFO
export NCCL_NVLS_ENABLE=0

export CFLAGS="-I/usr/include"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"
export CUTLASS_PATH="/path/to/cutlass"

deepspeed --hostfile=hostfile.txt lerobot/scripts/ddp_train.py \
    --deepspeed="./ds_zero2.json" \
    --policy.type="pi0" \
    --dataset.root="/mnt/wangxiaofa/robot_dataset/lerobot-format/bridge_orig_lerobot/" \
    --dataset.repo_id="whatever" \
    --output_dir="/mnt/wangxiaofa/pi_0_ckpts/0308_first_bridge" \
    --batch_size=4 \
    --wandb.enable=true \
    --wandb.project="pi0first" \
    --job_name="pi0_0308_2nd" \
    --save_freq=10000 \
    --log_dir="/mnt/wangxiaofa/logs"