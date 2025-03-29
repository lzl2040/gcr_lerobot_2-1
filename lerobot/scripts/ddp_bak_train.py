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
import logging
import time
import os
from pathlib import Path
from datetime import datetime
from contextlib import nullcontext
from pprint import pformat
from typing import Any

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

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
        self.log_file = Path(os.path.join(cfg.log_dir, f"logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
        self.log_file.parent.mkdir(exist_ok=True)
        
        # 主进程初始化日志文件
        if self.accelerator.is_main_process:
            with open(self.log_file, 'w') as f:
                f.write(f"Training Log - Start at {datetime.now()}\n")

    def log(self, message: str, level: str = "INFO"):
        """核心日志方法"""
        formatted = f"[{os.getpid()}]-[{datetime.now().strftime('%H:%M:%S')}]-[{level}] - {message}"
        
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
    accelerator: Accelerator,
    policy: PreTrainedPolicy,
    batch: Any,
    grad_clip_norm: float,
) -> tuple[MetricsTracker, dict]:
    
    policy.train()
    with accelerator.autocast():
        loss, output_dict = policy.forward(batch)

    accelerator.backward(loss)
    
    grad_norm = None
    
    if accelerator.sync_gradients:
        grad_norm = accelerator.clip_grad_norm_(
            policy.parameters(),
            grad_clip_norm,
        )
    return loss, output_dict, grad_norm


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()
    
    # Initialize Accelerator
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=True,
        static_graph=False
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        kwargs_handlers=[ddp_kwargs]
    )
    
    logger = AccelerateLogger(accelerator, cfg)
    
    if accelerator.is_main_process:
        logger.info(pformat(cfg.to_dict()))
        if cfg.wandb.enable and cfg.wandb.project:
            wandb_logger = WandBLogger(cfg)
        else:
            wandb_logger = None
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))
    else:
        wandb_logger = None

    if cfg.seed is not None:
        set_seed(cfg.seed + accelerator.process_index)  # Add process index for deterministic seeding

    # Dataset and policy setup
    dataset = make_dataset(cfg)
    logger.info(f"Dataset: {dataset}")

    # Policy setup
    logger.info("Creating policy...")
    policy = make_policy(
        cfg=cfg.policy,
        device=accelerator.device,
        ds_meta=dataset.meta,
    )

    # Environment setup (only in main process)
    eval_env = None
    if accelerator.is_main_process and cfg.eval_freq > 0 and cfg.env is not None:
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size)

    # Optimizer and scheduler
    logger.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # Prepare components with Accelerator
    policy, optimizer, lr_scheduler = accelerator.prepare(
        policy, optimizer, lr_scheduler
    )

    # Resume training state
    step = 0
    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(
            cfg.checkpoint_path, optimizer, lr_scheduler, accelerator
        )

    # Logging setup (main process only)
    if accelerator.is_main_process:
        num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        num_total_params = sum(p.numel() for p in policy.parameters())
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # Dataloader setup
    if hasattr(cfg.policy, "drop_n_last_frames"):
        sampler = DistEpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index
        )
    else:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=True,
            seed=cfg.seed
        )
        shuffle = False

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    dataloader = accelerator.prepare(dataloader)
    dl_iter = cycle(dataloader)

    # Metrics setup
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
        "optim_s": AverageMeter("optim_s", ":.3f"),
    }
    train_tracker = MetricsTracker(
        cfg.batch_size * accelerator.num_processes * cfg.gradient_accumulation_steps,  # Total batch size across all processes
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step
    )

    # Main training loop
    accelerator.wait_for_everyone()
    logger.info(f"Start training on {accelerator.num_processes} devices")
    total_steps = cfg.steps * cfg.gradient_accumulation_steps
    completed_steps = step * cfg.gradient_accumulation_steps
    for _ in range(completed_steps, total_steps):
        dataloading_s = 0
        start_time = time.perf_counter()
        batch = next(dl_iter)
        dataloading_time = time.perf_counter() - start_time
        dataloading_s += dataloading_time
        
        fwd_bwd = 0
        fwd_bwd_start = time.perf_counter()
        loss, output_dict, grad_norm = update_policy(
            accelerator,
            policy,
            batch,
            cfg.optimizer.grad_clip_norm,
        )
        fwd_bwd_time = time.perf_counter() - fwd_bwd_start
        fwd_bwd += fwd_bwd_time
        
        if accelerator.sync_gradients:
            train_tracker.dataloading_s = dataloading_s
            train_tracker.update_s = fwd_bwd
            
            fwd_bwd = 0
            dataloading_s = 0
            
            opt_step_start = time.perf_counter()
            optimizer.step()
            optimizer.zero_grad()
            
            if lr_scheduler is not None:
                lr_scheduler.step()
            
            opt_step_time = time.perf_counter() - opt_step_start
            train_tracker.optim_s = opt_step_time
            
            step += 1

            if has_method(policy, "update"):
                policy.update()
                
            loss_value = accelerator.gather(loss).mean().item()
            grad_norm_value = accelerator.gather(grad_norm).mean().item() if grad_norm is not None else None
            
            # update metrics
            train_tracker.loss = loss_value
            train_tracker.grad_norm = grad_norm_value
            train_tracker.lr = optimizer.param_groups[0]["lr"]
            train_tracker.step()
            
        completed_steps += 1
        
        

        # Logging and checkpointing (main process only)
        if accelerator.is_main_process:
            is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
            is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
            is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

            if is_log_step:
                logger.info(train_tracker)
                if wandb_logger:
                    wandb_log_dict = train_tracker.to_dict()
                    if output_dict:
                        wandb_log_dict.update(output_dict)
                    wandb_logger.log_dict(wandb_log_dict, step)
                train_tracker.reset_averages()

            if cfg.save_checkpoint and is_saving_step:
                logger.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(checkpoint_dir, step, cfg, accelerator.unwrap_model(policy), 
                              optimizer.optimizer, lr_scheduler.scheduler)
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            if cfg.env and is_eval_step:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                with torch.no_grad():
                    eval_info = eval_policy(
                        eval_env,
                        accelerator.unwrap_model(policy),
                        cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed + accelerator.process_index,
                    )

                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size * accelerator.num_processes,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step
                )
                eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
                eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
                eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
                logging.info(eval_tracker)
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")

    # Cleanup
    if accelerator.is_main_process and eval_env:
        eval_env.close()
    accelerator.wait_for_everyone()
    accelerator.end_training()
    logger.info("Training finished")


if __name__ == "__main__":
    os.environ['WANDB_API_KEY'] = '7f1c1acfe477063902c617b0e8ef24d2b76ed447'
    train()