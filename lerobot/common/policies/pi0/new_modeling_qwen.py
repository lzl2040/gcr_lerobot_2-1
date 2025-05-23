#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

"""
π0: A Vision-Language-Action Flow Model for General Robot Control

[Paper](https://www.physicalintelligence.company/download/pi0.pdf)
[Jax code](https://github.com/Physical-Intelligence/openpi)

Designed by Physical Intelligence. Ported from Jax by Hugging Face.

Install pi0 extra dependencies:
```bash
pip install -e ".[pi0]"
```

Example of finetuning the pi0 pretrained model (`pi0_base` in `openpi`):
```bash
python lerobot/scripts/train.py \
--policy.path=lerobot/pi0 \
--dataset.repo_id=danaaubakirova/koch_test
```

Example of finetuning the pi0 neural network with PaliGemma and expert Gemma
pretrained with VLM default parameters before pi0 finetuning:
```bash
python lerobot/scripts/train.py \
--policy.type=pi0 \
--dataset.repo_id=danaaubakirova/koch_test
```

Example of using the pi0 pretrained model outside LeRobot training framework:
```python
policy = Pi0Policy.from_pretrained("lerobot/pi0")
```

"""

import math
import numpy as np
from collections import deque
import os

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from transformers import AutoTokenizer, AutoProcessor

from qwen_vl_utils import process_vision_info
from PIL import Image

from lerobot.common.constants import ACTION, OBS_ROBOT
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pi0.configuration_qwen import QwenConfig
from lerobot.common.policies.pi0.qwen_with_expert import (
    PaliGemmaWithExpertConfig,
    PaliGemmaWithExpertModel,
)
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.utils.utils import get_safe_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def sample_beta(alpha, beta, bsize, device):
    gamma1 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / alpha)
    gamma2 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / beta)
    return gamma1 / (gamma1 + gamma2)


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def pad_vector(vector, new_dim):
    """Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    # This ensures that the input stays within
    # [−1,1] to avoid invalid values for arcsin
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with pi0 which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    # Convert from the gripper position used by pi0 to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)



class QwenPolicy(PreTrainedPolicy):
    """Wrapper class around PI0FlowMatching model to train and run inference within LeRobot."""

    config_class = QwenConfig
    name = "qwen"

    def __init__(
        self,
        config: QwenConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """

        super().__init__(config)
        config.validate_features()
        self.config = config
        
        self.processor = AutoProcessor.from_pretrained(config.qwen_path)
        self.processor.tokenizer.padding_side = "left"
        
        self.dtype = torch.bfloat16

        self.reset()
        if config.train_from_scratch:
            print(f"Training from scratch, loading qwen2.5 vl weights from {config.qwen_path}.")
            self.model = QwenFlowMatching(config, init_load=True, init_path = config.qwen_path)
        else:
            self.model = QwenFlowMatching(config, init_load=False)
        self.model.paligemma_with_expert.set_requires_grad()
        # gc_kwargs = {"use_reentrant": False}
        # self.model.paligemma_with_expert.qwen25vl.model.gradient_checkpointing = True
        # self.model.paligemma_with_expert.qwen25vl.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)
        # self.model.paligemma_with_expert.qwen_expert.model.gradient_checkpointing = True
        # self.model.paligemma_with_expert.qwen_expert.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)
        
    def init_load(self, path):
        """Load the model weights from a checkpoint."""
        self.model.init_load(path)

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def get_optim_params(self) -> dict:
        return self.parameters()

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()

        if self.config.adapt_to_pi_aloha:
            batch[OBS_ROBOT] = self._pi_aloha_decode_state(batch[OBS_ROBOT])

        # 先归一化，然后再pad
        # batch = self.normalize_inputs(batch)

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            images, img_masks = self.prepare_images(batch)
            state = self.prepare_state(batch)
            lang_tokens, lang_masks = self.prepare_language(batch)

            actions = self.model.sample_actions(
                images, img_masks, lang_tokens, lang_masks, state, noise=noise
            )

            # Unpad actions
            original_action_dim = self.config.action_feature.shape[0]
            actions = actions[:, :, :original_action_dim]

            # actions = self.unnormalize_outputs({"action": actions})["action"]

            if self.config.adapt_to_pi_aloha:
                actions = self._pi_aloha_encode_actions(actions)

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor], noise=None, time=None) -> tuple[Tensor, dict[str, Tensor]]:
        """Do a full training forward pass to compute the loss"""
        if self.config.adapt_to_pi_aloha:
            batch[OBS_ROBOT] = self._pi_aloha_decode_state(batch[OBS_ROBOT])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        # 先归一化，然后再pad
        # batch = self.normalize_inputs(batch)
        # batch = self.normalize_targets(batch)
        
        # input_ids, attention_mask, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, second_per_grid_ts = self.prepare_input(batch)
        
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        pixel_values = batch["pixel_values"]
        image_grid_thw = batch["image_grid_thw"]
        pixel_values_videos = batch["pixel_values_videos"]
        video_grid_thw = batch["video_grid_thw"]
        second_per_grid_ts = batch["second_per_grid_ts"]
        
        actions = self.prepare_action(batch)
        actions = self.convert_to_dtype(actions)
        actions_is_pad = batch.get("actions_id_pad")
        
        state = self.prepare_state(batch)
        state = self.convert_to_dtype(state)
        
        noise = self.convert_to_dtype(noise)
        time = self.convert_to_dtype(time)

        loss_dict = {}
        losses = self.model.forward(input_ids, attention_mask, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, second_per_grid_ts, state, actions, noise, time)
        loss_dict["losses_after_forward"] = losses.clone()

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone()

        # Remove padding
        losses = losses[:, :, : self.config.max_action_dim]
        loss_dict["losses_after_rm_padding"] = losses.clone()

        # For backward pass
        loss = losses.mean()
        # For logging
        loss_dict["l2_loss"] = loss.item()

        return loss, loss_dict
    
    def convert_to_dtype(self, vector:torch.Tensor):
        if not isinstance(vector, type(None)):
            if vector.is_floating_point():
                vector = vector.to(dtype=self.dtype)
        return vector

    def prepare_images(self, batch):
        """Apply Pi0 preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        
        visions = []
        device = None

        present_img_keys = [key for key in self.config.image_features if key in batch]
        # missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )
        # print(f"Present image keys: {present_img_keys}")
        for key in present_img_keys:
            img_seq = batch[key]
            # print(f"key: {key}, img_seq: {img_seq.shape}")
            # bsize = img_seq.shape[0]
            if isinstance(img_seq, list):
                bsize = len(img_seq)
            elif isinstance(img_seq, torch.Tensor):
                bsize = img_seq.shape[0]
            elif isinstance(img_seq, np.ndarray):
                bsize = img_seq.shape[0]
            else:
                raise ValueError(f"Unknown type for img_seq: {type(img_seq)}")
            break
        for i in range(bsize):
            vision = {
                "image": [],
                "video": None,
            }
            visions.append(vision)
        # Preprocess image features present in the batch
        for key in present_img_keys:
            img_seq = batch[key]
            # device = img_seq.device
            vision = {
                "image": [],
                "video": None,
            }
            # If the image sequence is a list, it means we have a video
            # 
            if key == "observation.images.primary":
                for i in range(bsize):
                    visions[i]['video'] = img_seq[i]
                    video_length = len(img_seq[i])
                    if video_length > self.config.max_frame:
                        # Sample the video from the ending
                        visions[i]["video"] = img_seq[i][-self.config.max_frame:]
            else:
                for i in range(bsize):
                    visions[i]['image'].append(img_seq[i][0])
            # if img_seq.ndim == 5:
            #     for i in range(bsize):
            #         video = []
            #         for j in range(img_seq[i].shape[0]):
            #             img = img_seq[i][j]
            #             img = img.cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
            #             img_pil = Image.fromarray(img).resize((112, 112))
            #             video.append(img_pil)
            #         video_length = batch['video_lengths'][i]
            #         # if int(os.environ.get("RANK", 0)) == 0:
            #         # print(f"video_length: {video_length}, config max frame: {self.config.max_frame}, frames sent in : {img_seq[i].shape[0]}")
                    
            #         visions[i]["video"] = video[:video_length]
            #         if video_length > self.config.max_frame:
            #             # Sample the video from the ending
            #             visions[i]["video"] = video[-self.config.max_frame:]
                        
            #         # if int(os.environ.get("RANK", 0)) == 0:
            #         current_video_length = len(visions[i]["video"])
            #         current_frame_size = visions[i]["video"][0].size
            #         # print(f"Video after preprocessing: {current_video_length}, {current_frame_size}")
            # else:
            #     for i in range(bsize):
            #         img = img_seq[i].cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
            #         img = Image.fromarray(img)
            #         visions[i]["image"].append(img)

        return visions, bsize, device

    def prepare_language(self, batch, visions) -> tuple[Tensor, Tensor]:
        """Tokenize the text input"""
        
        def apply_template(text, vision=None):
            message = [
                {"role": "user",
                "content": [
                    
                ],}
            ]
            if "video" in vision.keys():
                if vision["video"] is not None:
                    message[0]["content"].append(
                        {
                            "type": "video",
                            "video": vision["video"],
                        },
                    )
            for i in range(len(vision["image"])):
                message[0]["content"].append(
                    {
                        "type": "image",
                        "image": vision["image"][i],
                    }
                )
            message[0]["content"].append({"type": "text", "text": text})
            return self.processor.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
        # device = batch[OBS_ROBOT].device
        tasks = batch["task"]

        # Qwen2.5VL prompt uses a chat template
        templates = []
        for index in range(len(tasks)):
            templates.append(apply_template(tasks[index], visions[index]))
        
        return templates
    
    def prepare_input(self, batch):
        """Prepare the input for the model"""
        
        visions, bsize, device = self.prepare_images(batch)
        if device is None:
            device = self.model.paligemma_with_expert.qwen25vl.device
        tasks = self.prepare_language(batch, visions)
        texts = batch["task"]
        messages = []
        
        for i in range(bsize):
            video = visions[i]["video"]
            message = [{
                "role": "user",
                "content": []
            }]
            if video is not None:
                message[0]["content"].append(
                    {
                        "type": "video",
                        "video": video,
                    },
                )
            for j in range(len(visions[i]["image"])):
                message[0]["content"].append(
                    {
                        "type": "image",
                        "image": visions[i]["image"][j],
                    }
                )
            message[0]["content"].append(
                {"type": "text", "text": texts[i]}
            )
            messages.append(message)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        # if int(os.environ.get("RANK", 0)) == 0:
        # if video_inputs is not None:
        #     video_lengths = [len(video) for video in video_inputs]
        #     print(f"Num of video: {len(video_inputs)}, Length per video: {video_lengths}, Video frame size: {video_inputs[0][0].size}, {len(video_inputs[0][0].split())}")
        # print(f"image_inputs: {len(image_inputs)}, {image_inputs[0].size}")
        inputs = self.processor(
            text=tasks,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        input_ids = getattr(inputs, "input_ids", None)
        attention_mask = getattr(inputs, "attention_mask", None)
        pixel_values = getattr(inputs, "pixel_values", None)
        image_grid_thw = getattr(inputs, "image_grid_thw", None)
        pixel_values_videos = getattr(inputs, "pixel_values_videos", None)
        video_grid_thw = getattr(inputs, "video_grid_thw", None)
        second_per_grid_ts = getattr(inputs, "second_per_grid_ts", None)
        
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=device, dtype=self.dtype)
        if pixel_values is not None:
            pixel_values = pixel_values.to(device=device, dtype=self.dtype)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(device=device, dtype=self.dtype)
        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.to(device=device, dtype=self.dtype)
        if video_grid_thw is not None:
            video_grid_thw = video_grid_thw.to(device=device, dtype=self.dtype)
        
        return input_ids, attention_mask, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, second_per_grid_ts
        
        # return inputs
        

    def _pi_aloha_decode_state(self, state):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        # Flip the joints again.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions

    def prepare_state(self, batch):
        """Pad state"""
        state = pad_vector(batch[OBS_ROBOT], self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions


class QwenFlowMatching(nn.Module):
    """
    π0: A Vision-Language-Action Flow Model for General Robot Control

    [Paper](https://www.physicalintelligence.company/download/pi0.pdf)
    [Jax code](https://github.com/Physical-Intelligence/openpi)

    Designed by Physical Intelligence. Ported from Jax by Hugging Face.
    ┌──────────────────────────────┐
    │               actions        │
    │               ▲              │
    │              ┌┴─────┐        │
    │  kv cache    │Gemma │        │
    │  ┌──────────►│Expert│        │
    │  │           │      │        │
    │ ┌┴────────┐  │x 10  │        │
    │ │         │  └▲──▲──┘        │
    │ │PaliGemma│   │  │           │
    │ │         │   │  robot state │
    │ │         │   noise          │
    │ └▲──▲─────┘                  │
    │  │  │                        │
    │  │  image(s)                 │
    │  language tokens             │
    └──────────────────────────────┘
    """

    def __init__(self, config, init_load = False, init_path = None):
        super().__init__()
        self.config = config
        
        self.dtype = torch.bfloat16

        paligemma_with_export_config = PaliGemmaWithExpertConfig(
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            attention_implementation=self.config.attention_implementation,
            train_main_layers=self.config.train_main_layers,
        )
        self.paligemma_with_expert = PaliGemmaWithExpertModel(paligemma_with_export_config, init_load = init_load, init_path = init_path)

        # Projections are float32
        state_proj_width = paligemma_with_export_config.awa_model_config["hidden_size"]
        # self.state_proj = nn.Linear(self.config.max_state_dim, self.config.proj_width)
        self.state_proj = nn.Linear(self.config.max_state_dim, state_proj_width)
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.config.proj_width)
        self.action_out_proj = nn.Linear(self.config.proj_width, self.config.max_action_dim)

        self.action_time_mlp_in = nn.Linear(self.config.proj_width * 2, self.config.proj_width)
        self.action_time_mlp_out = nn.Linear(self.config.proj_width, self.config.proj_width)

        self.set_requires_grad()
        
    def init_load(self, path):
        """Load the model weights from pretrained parameters."""
        self.paligemma_with_expert.init_load(path)

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=self.dtype,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=self.dtype, device=device)

    def embed_prefix(
        self, input_ids, attention_mask, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, second_per_grid_ts, position_ids=None, cache_position=None, rope_deltas=None, past_key_values=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        input_ids = input_ids.to(device=self.paligemma_with_expert.qwen25vl.device)
        inputs_embeds = self.paligemma_with_expert.qwen25vl.model.embed_tokens(input_ids)
        
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.dtype)
            image_grid_thw = image_grid_thw.type(torch.int32)
            image_embeds = self.paligemma_with_expert.custom_visual_forward(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = (input_ids == self.paligemma_with_expert.qwen25vl.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            
            if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

            mask = input_ids == self.paligemma_with_expert.qwen25vl.config.image_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)
            
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            
        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.dtype)
            video_grid_thw = video_grid_thw.type(torch.int32)
            # if int(os.environ.get("RANK", 0)) == 0:
            # print(f"video_grid_thw: {video_grid_thw.shape}")
            # print(f"pixel_values_videos: {pixel_values_videos.shape}")
            video_embeds = self.paligemma_with_expert.custom_visual_forward(pixel_values_videos, grid_thw=video_grid_thw)
            n_video_tokens = (input_ids == self.paligemma_with_expert.qwen25vl.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            
            if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

            mask = input_ids == self.paligemma_with_expert.qwen25vl.config.video_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            video_mask = mask_expanded.to(inputs_embeds.device)
            
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
        
        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)
            
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.paligemma_with_expert.qwen25vl.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                # print(f"input_ids: {input_ids.shape}, image_grid_thw: {image_grid_thw.shape}, video_grid_thw: {video_grid_thw.shape}, second_per_grid_ts: {len(second_per_grid_ts)}-{second_per_grid_ts}, attention_mask: {attention_mask.shape}")
                position_ids, rope_deltas = self.paligemma_with_expert.qwen25vl.get_rope_index(
                    input_ids = input_ids,
                    image_grid_thw = image_grid_thw,
                    video_grid_thw = video_grid_thw,
                    second_per_grid_ts = second_per_grid_ts,
                    attention_mask = attention_mask,
                )
                self.paligemma_with_expert.qwen25vl.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.paligemma_with_expert.qwen25vl.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        
        # if attention_mask is not None:
        #     if attention_mask.ndim == 2:
        #         for i in range(attention_mask.shape[0]):
        #             for j in range(attention_mask.shape[1]):
        #                 if attention_mask[i][j] == 1:
        #                     attention_mask[i][j] = 0
        
        return inputs_embeds, attention_mask, position_ids
                

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []
        
        # print("state dtype: ", state.dtype)
        # print("noise_action_dtype: ", noisy_actions.dtype)
        # print("time_step dtype: ", timestep.dtype)

        # Embed state
        state_emb = self.state_proj(state)
        state_emb = state_emb.to(dtype=self.dtype)
        embs.append(state_emb[:, None, :])
        bsize = state_emb.shape[0]
        dtype = state_emb.dtype
        device = state_emb.device

        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.config.proj_width, min_period=4e-3, max_period=4.0, device=device
        )
        time_emb = time_emb.type(dtype=dtype)

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([1] * (self.config.n_action_steps - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def forward(
        self, input_ids, attention_mask, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, second_per_grid_ts, state, actions, noise=None, time=None
    ) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_att_masks, prefix_pos_ids = self.embed_prefix(
            input_ids, attention_mask, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, second_per_grid_ts
        )
        
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, time)

        # pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        # print(f"init att masks: {att_masks.shape}")

        # att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        # position_ids = torch.cumsum(pad_masks, dim=1) - 1
        
        # print(f"Prefix Emb shape: {prefix_embs.shape}")
        # print(f"Suffix Emb shape: {suffix_embs.shape}")

        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_masks,
            position_ids=prefix_pos_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs[:, 0:1, :], suffix_embs[:, 1:, :]],
            use_cache=True,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        # Original openpi code, upcast attention output
        # suffix_out = suffix_out.to(dtype=torch.float32)
        suffix_out = suffix_out.to(dtype=self.dtype)
        v_t = self.action_out_proj(suffix_out)

        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.n_action_steps, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step
            x_t += dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t
