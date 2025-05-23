torchrun --nnodes=1 \
    --nproc_per_node=2 \
    lerobot/scripts/fsdp_train.py \
    --policy.type="qwen" \
    --output_dir="/data_16T/deepseek/" \
    --save_freq=10000 \
    --dataset.repo_id="whatever" \
    --dataset.processor="/datassd_1T/qwen25vl/Qwen2.5-VL-7B-Instruct/" \
    --dataset.parent_dir="/data_16T/lerobot_openx/" \
    --output_dir="/data_16T/deepseek/qwen_flow/" \
    --policy.scheduler_warmup_steps=1000 \
    --policy.scheduler_platform_steps=10000 \
    --policy.scheduler_decay_steps=100000 \
    --policy.optimizer_lr=1e-3 \
    --policy.train_main_layers=0 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false
    

