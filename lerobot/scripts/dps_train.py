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
import json
from pathlib import Path
from datetime import datetime
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import deepspeed
from deepspeed import get_accelerator

import torch
from termcolor import colored
from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.transforms import ImageTransforms
from lerobot.common.datasets.lerobot_dataset import MultiDatasetforDistTraining, extra_collate_fn
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

def init_logger(cfg):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    if cfg.local_rank == 0:
        formatter = logging.Formatter(
            f'[%(asctime)s] [rank: {cfg.local_rank}] [%(levelname)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # # 控制台Handler
        # console_handler = logging.StreamHandler()
        # console_handler.setFormatter(formatter)
        # logger.addHandler(console_handler)
        
        # 文件Handler
        log_path = Path(cfg.log_dir) / f"logs_with_pretrain/{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def rank_dataloader_check(model_engine, batch):
    batch = {k: v.to(model_engine.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}, {value.dtype}, {value.device}")
        elif isinstance(value, list):
            print(f"{key}: {len(value)}, example: {value[0]}")
        else:
            print(f"{key}: {value}")
            
def update_policy(
    model_engine,
    batch: Any,
    logger
) -> tuple[MetricsTracker, dict]:
    
    batch = {k: v.to(model_engine.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    # torch.cuda.empty_cache()
    loss, output_dict = model_engine(batch)

    model_engine.backward(loss)
    
    # for name, param in model_engine.module.model.paligemma_with_expert.qwen25vl.model.layers[-1].named_parameters():
    #     if param.grad is not None:
    #         print(f"{name} gradient norm: {param.grad.norm().item()}")
    
    model_engine.step()
    
   # torch.cuda.empty_cache()
    return loss, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()
    
    # Initialize DeepSpeed
    deepspeed.init_distributed()
    logger = init_logger(cfg)
    
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms)
    )
    
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(pformat(cfg.to_dict()))
        if cfg.wandb.enable and cfg.wandb.project:
            wandb_logger = WandBLogger(cfg)
        else:
            wandb_logger = None
            logger.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))
    else:
        wandb_logger = None

    if cfg.seed is not None:
        set_seed(cfg.seed + int(os.environ.get('RANK', 0)))

    # Dataset setup
    dataset = MultiDatasetforDistTraining(cfg=cfg, image_transforms=image_transforms, 
                           seed=cfg.seed + int(os.environ.get("RANK", 0)), 
                           data_mix="oxe_magic_soup_plus",
                           vla2root_json="vla2root_bak_single.json",
                        #    vla2root_json="vla2root.json"
                           )
    # dataset = MultiDatasetforDistTraining(cfg=cfg, image_transforms=image_transforms, 
    #                        seed=cfg.seed + int(os.environ.get("RANK", 0)), data_mix="oxe_magic_soup_plus",
    #                        vla2root_json="vla2root_bak_single.json")
    logger.info(f"Dataset: {dataset}")

    # Policy setup
    logger.info("Creating policy...")
    if hasattr(cfg.policy, "tokenizer_max_length"):
        logger.info("Setting model's tokenizer_max_length to 100")
        cfg.policy.tokenizer_max_length=100
    logger.info("Still creating policy...")
    # print(cfg.policy.pretrained_path)
    policy = make_policy(
        cfg=cfg.policy,
        device='cpu',
        ds_meta=dataset.meta,
        # weight_pt_path="/mnt/wangxiaofa/pi0_pretrain/model.pt"
    )
    logger.info("Policy model created...")

    # Environment setup (only in main process)
    eval_env = None
    if int(os.environ.get('RANK', 0)) == 0 and cfg.eval_freq > 0 and cfg.env is not None:
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size)

    # Optimizer and scheduler
    logger.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # Logging setup (main process only)
    if int(os.environ.get('RANK', 0)) == 0:
        num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        num_total_params = sum(p.numel() for p in policy.parameters())
        logger.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logger.info(f"{cfg.env.task=}")
        logger.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logger.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logger.info(f"{dataset.num_episodes=}")
        logger.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logger.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # Dataloader setup
    if hasattr(cfg.policy, "drop_n_last_frames"):
        sampler = DistEpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
            num_replicas=int(os.environ.get('WORLD_SIZE', 1)),
            rank=int(os.environ.get('RANK', 0))
        )
    else:
        logger.info("Creating DistributedSampler")
        sampler = DistributedSampler(
            dataset,
            num_replicas=int(os.environ.get('WORLD_SIZE', 1)),
            rank=int(os.environ.get('RANK', 0)),
            shuffle=True,
            seed=cfg.seed
        )

    with open(cfg.deepspeed, 'r') as f:
        deepspeed_configs_in_dict = json.load(f)
    batch_size = deepspeed_configs_in_dict['train_micro_batch_size_per_gpu']
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            sampler=sampler,
                            num_workers=4,
                            pin_memory=True,
                            collate_fn=extra_collate_fn,
                            # persistent_workers=True,
                            # prefetch_factor=2
                            )
    # DeepSpeed initialization
    # model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
    #     model=policy,
    #     optimizer=optimizer,
    #     lr_scheduler=lr_scheduler,
    #     config=cfg.deepspeed,
    #     model_parameters=policy.parameters(),
    # )
    params = list(policy.model.paligemma_with_expert.qwen_expert.parameters()) + list(policy.model.paligemma_with_expert.qwen25vl.model.parameters())
    # params = list(policy.model.paligemma_with_expert.qwen_expert.parameters())
    params = list(filter(lambda p: p.requires_grad, params))
    
    # Resume training state
    step = 0
    if cfg.resume:
        logger.info(f"Resuming training from {cfg.output_dir}")
        ckpt_path = cfg.output_dir
        ckpt_list = os.listdir(ckpt_path)
        latest_ckpt = sorted(ckpt_list, key=lambda x: int(x.split("step")[-1]))[-1]
        checkpoint_path = os.path.join(ckpt_path, latest_ckpt)
        step = int(latest_ckpt.split("step")[-1])
        
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        policy.load_state_dict(state_dict, strict=True)
        
        logger.info(f"Resumed training from step {step}")
    else:
        client_state = {
            'step': step
        }
    
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=policy,
        config=cfg.deepspeed,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler
    )
    
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(model_engine)
        logger.info(model_engine.module)
   
    logger.info(f"Training batch size:{model_engine.train_batch_size()}") # micro_size * gradient_cum_size * gpu_num
        
    dl_iter = cycle(dataloader)
    
    for i in range(5):
        batch = next(dl_iter)
        rank_dataloader_check(model_engine, batch)

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
        model_engine.train_batch_size(),
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=int(step/model_engine.gradient_accumulation_steps())
    )

    # Main training loop
    logger.info(f"Start training on {int(os.environ.get('WORLD_SIZE', 1))} devices")
    total_steps = cfg.steps
    completed_steps = step
    fwd_bwd_time = 0
    dataloading_s = 0
    dist_step=10
    
    for step_idx in range(completed_steps, total_steps):
        
        
        start_time = time.perf_counter()
        batch = next(dl_iter)
        dataloading_time = time.perf_counter() - start_time
        dataloading_s += dataloading_time
        
        # print(batch['observation.images.primary'].shape)
        # print(batch['observation.images.secondary'].shape)
        # print(batch['observation.images.wrist'].shape)
        
        fwd_bwd_start = time.perf_counter()
        loss, output_dict = update_policy(
            model_engine,
            batch,
            logger
        )
        step += 1
        fwd_bwd_time += time.perf_counter() - fwd_bwd_start
        
        if model_engine.is_gradient_accumulation_boundary():
            train_tracker.dataloading_s = dataloading_s
            train_tracker.update_s = fwd_bwd_time
            

            loss_value = loss.detach().mean().item()
            grad_norm_value = 0.0
            
            train_tracker.loss = loss_value
            train_tracker.grad_norm = grad_norm_value
            train_tracker.lr = optimizer.param_groups[0]["lr"]
            train_tracker.step()
            
            fwd_bwd_time=0
            dataloading_s=0
        
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0
        
        if cfg.save_checkpoint and is_saving_step:
            logger.info(f"Checkpoint policy after step {step}")
            # checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            os.makedirs(cfg.output_dir, exist_ok=True)
            
            client_state['step'] = step
            if int(os.environ.get('RANK', 0)) == 0:
                torch.save(model_engine.module.state_dict(), os.path.join(cfg.output_dir, f"step{step}.pt"))
            dist.barrier(device_ids=[model_engine.local_rank])
            # model_engine.save_checkpoint(
            #     save_dir=cfg.output_dir,
            #     client_state=client_state
            # )
            # torch.save(client_state, os.path.join(checkpoint_dir, "metadata.pt"))
            # update_last_checkpoint(checkpoint_dir)
        
        if int(os.environ.get('RANK', 0)) == 0:
            if is_log_step:
                logger.info(train_tracker)
                if wandb_logger:
                    wandb_log_dict = train_tracker.to_dict()
                    if output_dict:
                        wandb_log_dict.update(output_dict)
                    wandb_logger.log_dict(wandb_log_dict, step)
                train_tracker.reset_averages()

            if cfg.env and is_eval_step:
                torch.cuda.empty_cache()
                step_id = get_step_identifier(step, cfg.steps)
                logger.info(f"Eval policy at step {step}")
                with torch.no_grad():
                    eval_info = eval_policy(
                        eval_env,
                        model_engine.module,
                        cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed + int(os.environ.get('RANK', 0)),
                    )

                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size * int(os.environ.get('WORLD_SIZE', 1)),
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step
                )
                eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
                eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
                eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
                logger.info(eval_tracker)
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")
        if step_idx % dist_step == 0:
            dist.barrier(device_ids=[model_engine.local_rank])
    # Cleanup
    if int(os.environ.get('RANK', 0)) == 0 and eval_env:
        eval_env.close()
    logger.info("Training finished")


if __name__ == "__main__":
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['WANDB_API_KEY'] = '7f1c1acfe477063902c617b0e8ef24d2b76ed447'
    train()