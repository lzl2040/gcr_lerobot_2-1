#!/bin/bash

# 多机配置
# MASTER_ADDR=${MASTER_NODE_IP}  # 主节点IP
MASTER_PORT=29500
NUM_NODES=1
NUM_PROCESSES_PER_NODE=2

accelerate launch \
    --config_file accelerate_config.yaml \
    --num_processes $((NUM_NODES * NUM_PROCESSES_PER_NODE)) \
    --num_machines $NUM_NODES \
    --machine_rank 0 \
    --main_process_port $MASTER_PORT \
    lerobot/scripts/ds_train.py \
    --policy.type="pi0" \
    --dataset.root="/data_16T/lerobot_openx/fmb_dataset_lerobot/" \
    --dataset.repo_id="whatever" \
    --output_dir="/data_16T/deepseek/pi_1" \
    --batch_size=4 \
    --wandb.enable=true \
    --wandb.project="pi0first" \
    --job_name="pi0_on_fractal" \
    --save_freq=20