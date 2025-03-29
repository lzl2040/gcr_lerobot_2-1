#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import logging
import os
import copy
import sys
from pathlib import Path
from datetime import datetime
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
import torch.distributed as dist
from termcolor import colored
from torch.optim import Optimizer
from collections import OrderedDict as orderdict

# Accelerate集成
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.state import PartialState
from accelerate.utils import DistributedType

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler, DistEpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
    StateChunkDataset,
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy

class AccelerateLogger:
    def __init__(self, accelerator: Accelerator, cfg):
        self.accelerator = accelerator
        self.log_file = Path(os.path.join(cfg.log_dir,f"logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
        self.log_file.parent.mkdir(exist_ok=True)
        
        # 主进程初始化日志文件
        if self.accelerator.is_main_process:
            with open(self.log_file, 'w') as f:
                f.write(f"Training Log - Start at {datetime.now()}\n")

    def log(self, message: str, level: str = "INFO"):
        """核心日志方法"""
        formatted = f"[{datetime.now().strftime('%H:%M:%S')}] [{level}] - {message}"
        
        # 主进程输出到控制台和文件
        if self.accelerator.is_main_process:
            print(formatted)
            with open(self.log_file, 'a') as f:
                f.write(formatted + "\n")
        else:
            # 其他进程仅打印
            self.accelerator.print(formatted)

    def critical(self, message: str):
        self.log(message, "CRITICAL")
        self.accelerator.fatal_error()

    def error(self, message: str):
        self.log(message, "ERROR")

    def warning(self, message: str):
        self.log(message, "WARNING")

    def info(self, message: str):
        self.log(message, "INFO")

    def debug(self, message: str):
        self.log(message, "DEBUG")


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    policy.train()
    
    # 混合精度上下文
    with accelerator.autocast():
        # print(batch)
        for _, value in batch.items():
            if isinstance(value, torch.Tensor):
                if value.is_floating_point() and value.dtype != torch.bfloat16:
                    value = value.to(dtype=torch.bfloat16)
        loss, output_dict = policy.forward(batch)
        
    # 反向传播
    accelerator.backward(loss)

    # 梯度裁剪
    if accelerator.sync_gradients:
        accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    
    optimizer.step()
    optimizer.zero_grad()

    # 学习率调度
    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        policy.update()

    # 收集分布式指标
    loss_value = accelerator.gather(loss).mean().item()
    grad_norm = accelerator.clip_grad_norm_(policy.parameters(), 1.0)
    
    train_metrics.loss = loss_value
    train_metrics.grad_norm = grad_norm.item().to(dtype=torch.float32) if grad_norm is not None else 0.0
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    
    # 初始化accelerator（兼容deepspeed启动方式）
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="bf16",
        log_with="wandb" if cfg.wandb.enable else None,
        kwargs_handlers=[
            DistributedDataParallelKwargs(
                find_unused_parameters=True,
                # bucket_cap_mb=cfg.distributed.bucket_cap_mb,
            )
        ]
    )
    
    cfg.validate()
    
    logger = AccelerateLogger(accelerator, cfg)
    
    if accelerator.is_local_main_process:
        logger.info("Accelerate is the best")
    if accelerator.is_main_process:
        logger.warning("No, it's not.")
    
    # 日志初始化（仅主进程）
    if accelerator.is_main_process:
        
        logger.info(pformat(cfg.to_dict()))
        if cfg.wandb.enable and cfg.wandb.project:
            wandb_logger = WandBLogger(cfg)
            accelerator.init_trackers("train")
        else:
            wandb_logger = None
            logger.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))
    else:
        wandb_logger = None

    if cfg.seed is not None:
        set_seed(cfg.seed + accelerator.process_index)  # 确保不同进程有不同的随机种子

    # 设备管理
    device = accelerator.device

    # 数据集初始化
    logger.info("Creating dataset")

    dataset = make_dataset(cfg)

    # 评估环境初始化（仅主进程）
    eval_env = None
    if accelerator.is_main_process and cfg.eval_freq > 0 and cfg.env is not None:
        logger.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size)

    # 策略初始化
    logger.info("Creating policy")
    if hasattr(cfg.policy, "tokenizer_max_length"):
        logger.info("Setiing model's tokenizer_max_length to 60")
        cfg.policy.tokenizer_max_length=60
    policy = make_policy(
        cfg=cfg.policy,
        device=device,
        ds_meta=dataset.meta,
    )
    policy = policy.to(dtype=torch.bfloat16)

    # 优化器初始化
    logger.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # 断点续训
    step = 0
    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(
            cfg.checkpoint_path, optimizer, lr_scheduler
        )

    # 分布式数据加载
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = DistEpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
        )
    else:
        from torch.utils.data import DistributedSampler
        shuffle = True
        # sampler = None
        sampler = DistributedSampler(
            dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=shuffle,  # 从配置读取shuffle选项
            seed=cfg.seed
        )
        shuffle = False
        

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=True,
        drop_last=False,
    )

    # 准备分布式组件
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    dl_iter = cycle(dataloader)

    # 训练指标初始化
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )

    # 训练主循环
    accelerator.wait_for_everyone()
    logger.info(f"Start training on {accelerator.num_processes} devices")
    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        # 策略更新
        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator,
            lr_scheduler=lr_scheduler,
        )

        step += 1
        train_tracker.step()
        
        # 日志记录周期
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0
        

        # 日志记录（主进程）
        if accelerator.is_main_process and is_log_step:
            logger.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()

                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()
            
        def offload_to_cpu(obj):
            if isinstance(obj, torch.Tensor):
                return obj.cpu()
            elif isinstance(obj, dict):
                return {k: offload_to_cpu(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [offload_to_cpu(v) for v in obj]
            elif isinstance(obj, tuple):
                return tuple(offload_to_cpu(v) for v in obj)
            else:
                return obj

        # 模型保存（主进程）
        if cfg.save_checkpoint and is_saving_step:
            logger.info(f"Merging base optim state at step {step}")
            local_state = optimizer.optimizer.state_dict()["base_optimizer_state"]
            
            local_state_cpu = offload_to_cpu(local_state)
            
            del local_state
            torch.cuda.empty_cache()
            
            gathered_states = [None] * dist.get_world_size()
            dist.gather_object(local_state_cpu, gathered_states if accelerator.is_main_process else None, dst=0)
            
            del local_state_cpu
            torch.cuda.empty_cache()
            
            gathered_states = offload_to_cpu(gathered_states)
            torch.cuda.empty_cache()
            
            if accelerator.is_main_process:
                merged_state = {}
                for state_shard in gathered_states:
                    for param_id, param_state in state_shard.items():
                        merged_state[param_id] = {
                            k: v for k, v in param_state.items()
                        }
            del gathered_states
            torch.cuda.empty_cache()
            
        dist.barrier()
        
        if cfg.save_checkpoint and is_saving_step:
            logger.info(f"Merging param groups at step {step}")
            local_groups = optimizer.state_dict()["single_partition_of_fp32_groups"]
            local_groups_cpu = offload_to_cpu(local_groups)
            
            del local_groups
            torch.cuda.empty_cache()
            
            gathered_groups = [None] * dist.get_world_size()
            dist.gather_object(local_groups_cpu, gathered_groups if accelerator.is_main_process else None, dst=0)
            
            del local_groups_cpu
            torch.cuda.empty_cache()
            
            gathered_groups = offload_to_cpu(gathered_groups)
            torch.cuda.empty_cache()
            
            if accelerator.is_main_process:
                merged_groups = []
                for group_idx in range(len(gathered_groups[0])):
                    merged_group = torch.cat(
                        [shard[group_idx] for shard in gathered_groups], dim=0
                        )
                    merged_groups.append(merged_group)
            del gathered_groups
            torch.cuda.empty_cache()
        
        dist.barrier() # 确保所有进程同步
        if cfg.save_checkpoint and is_saving_step and accelerator.is_main_process:
            
            unwrapped_policy = accelerator.unwrap_model(policy)
            ori_optimizer, ori_lr_scheduler = make_optimizer_and_scheduler(cfg, unwrapped_policy)
            original_param_groups = ori_optimizer.param_groups
            del ori_lr_scheduler 
            del ori_optimizer
            torch.cuda.empty_cache()
            
            legacy_param_groups = []
            for group, merged_params in zip(original_param_groups, merged_groups):
                legacy_group = {
                    "lr": group["lr"],
                    "betas": group["betas"],
                    "eps": group["eps"],
                    "weight_decay": group["weight_decay"],
                    "amsgrad": group["amsgrad"],
                    "params": [param.data_ptr() for param in merged_params]
                }
                legacy_param_groups.append(legacy_group)

            merged_optim_state_dict = {
                "state": merged_state,
                "param_groups": legacy_param_groups
            }
 
            logger.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                    
            unwrapped_lr_scheduler = lr_scheduler.scheduler
            
            save_checkpoint(checkpoint_dir, step, cfg, unwrapped_policy, merged_optim_state_dict, unwrapped_lr_scheduler)
            update_last_checkpoint(checkpoint_dir)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)
        accelerator.wait_for_everyone()  # 确保所有进程同步

        # 模型评估（主进程）
        if cfg.env and is_eval_step and accelerator.is_main_process:
            step_id = get_step_identifier(step, cfg.steps)
            logger.info(f"Eval policy at step {step}")
            
            unwrapped_policy = accelerator.unwrap_model(policy)
            with torch.no_grad():
                eval_info = eval_policy(
                    eval_env,
                    unwrapped_policy,
                    cfg.eval.n_episodes,
                    videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                    max_episodes_rendered=4,
                    start_seed=cfg.seed,
                )

            eval_metrics = {
                "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                "pc_success": AverageMeter("success", ":.1f"),
                "eval_s": AverageMeter("eval_s", ":.3f"),
            }
            eval_tracker = MetricsTracker(
                cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step
            )
            eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
            eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
            eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
            
            logger.info(eval_tracker)
            if wandb_logger:
                wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")
        
        accelerator.wait_for_everyone()  # 确保所有进程同步
        
    # 清理资源
    if accelerator.is_main_process and eval_env:
        eval_env.close()
    accelerator.end_training()
    logger.info("Training completed")


if __name__ == "__main__":

    os.environ['WANDB_API_KEY'] = '7f1c1acfe477063902c617b0e8ef24d2b76ed447'
    train()