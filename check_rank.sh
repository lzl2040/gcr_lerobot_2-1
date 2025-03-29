#!/bin/bash
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=en,eth,em,bond,ib #bond0
export NCCL_DEBUG=INFO
export NCCL_NVLS_ENABLE=0

export CFLAGS="-I/usr/include"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"
export CUTLASS_PATH="/path/to/cutlass"

# export dataset_root="/data_16T/lerobot_openx/bridge_orig_lerobot/"
export dataset_root="/mnt/wangxiaofa/robot_dataset/lerobot-format/bridge_orig_lerobot/"

mpirun $$SINGULARITY_MPI_ENV \
    python lerobot/scripts/check_rank.py \
    --deepspeed="./ds_zero2.json" \
    --policy.type="pi0" \
    --dataset.root=$dataset_root \
    --dataset.repo_id="whatever" \
    --log_dir="/mnt/wangxiaofa/logs"
    # --output_dir="/mnt/wangxiaofa/pi_0_ckpts/rrrrrr" \
    # --batch_size=4