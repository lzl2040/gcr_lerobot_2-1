from typing import List, Optional, Union

import torch
import torch.version
from pytest import Cache
from torch import nn
from transformers import (
    AutoConfig,
    GemmaForCausalLM,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
    Qwen2ForCausalLM,
    PaliGemmaForConditionalGeneration,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.models.auto import CONFIG_MAPPING

from lerobot.common.policies.pi0.flex_attention import flex_attention_forward


def apply_rope(x, positions, max_wavelength=10_000):
    """
    Applies RoPE positions [B, L] to x [B, L, H, D].
    """
    d_half = x.shape[-1] // 2
    device = x.device
    dtype = x.dtype
    x = x.to(torch.float32)

    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=device)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)

    radians = radians[..., None, :]

    sin = torch.sin(radians)  # .to(dtype=dtype)
    cos = torch.cos(radians)  # .to(dtype=dtype)

    x1, x2 = x.split(d_half, dim=-1)
    res = torch.empty_like(x)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x2 * cos + x1 * sin

    return res.to(dtype)


class PaliGemmaWithExpertConfig(PretrainedConfig):
    model_type = "PaliGemmaWithExpertModel"
    sub_configs = {"paligemma_config": AutoConfig, "gemma_expert_config": AutoConfig, "qwen25vl_config": AutoConfig, "qwenexp_config": AutoConfig}

    def __init__(
        self,
        paligemma_config: dict | None = None,
        gemma_expert_config: dict | None = None,
        freeze_vision_encoder: bool = True,
        train_expert_only: bool = True,
        attention_implementation: str = "eager",
        qwen25vl_config: dict | None = None,
        qwenexp_config: dict | None = None,
        **kwargs,
    ):
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.attention_implementation = attention_implementation
        
        if qwen25vl_config is None:
            # Default config for qwen2_5vl
            self.qwen25vl_config = CONFIG_MAPPING["qwen2_5_vl"](
                transformers_version = "4.41.2",
                vocab_size=152064,
                bos_token_id=151643,
                eos_token_id=151645,
                hidden_size=3584,
                image_token_id=151655,
                video_token_id=151656,
                vision_start_token_id=151652,
                vision_end_token_id=151653,
                attention_dropout=0.0,
                hidden_act="silu",
                intermediate_size=18944,
                initializer_range=0.02,
                max_position_embeddings=128000,
                model_type="qwen2_5_vl",
                max_window_layers=28,
                num_attention_heads=28,
                num_hidden_layers=28,
                num_key_value_heads=4,
                rms_norm_eps=1e-06,
                rope_theta=1000000.0,
                sliding_window=32768,
                tie_word_embeddings=False,
                torch_dtype="bfloat16",
                use_cache=True,
                use_sliding_window=False,
                vision_config={
                    "depth": 32,
                    "hidden_act": "silu",
                    "hidden_size": 1280,
                    "intermediate_size": 3420,
                    "num_heads": 16,
                    "in_chans": 3,
                    "out_hidden_size": 3584,
                    "patch_size": 14,
                    "spatial_merge_size": 2,
                    "spatial_patch_size": 14,
                    "window_size": 112,
                    "fullatt_block_indexes": [
                        7,
                        15,
                        23,
                        31
                    ],
                    "tokens_per_second": 2,
                    "temporal_patch_size": 2
                },
                rope_scaling={
                    "type": "mrope",
                    "mrope_section": [
                        16,
                        24,
                        24
                    ]
                }
            )
        elif isinstance(self.qwen25vl_config, dict):
            # Override Pi0 default config for PaliGemma
            if "model_type" not in qwen25vl_config:
                qwen25vl_config["model_type"] = "qwen2_5_vl"

            cfg_cls = CONFIG_MAPPING[qwen25vl_config["model_type"]]
            self.qwen25vl_config = cfg_cls(**qwen25vl_config)
        
        if qwenexp_config is None:
            # Default config for qwen2_5vl
            self.qwenexp_config = CONFIG_MAPPING["qwen2"](
                transformers_version = "4.40.1",
                vocab_size=151936,
                bos_token_id=151643,
                eos_token_id=151643,
                hidden_size=1536,
                attention_dropout=0.0,
                hidden_act="silu",
                intermediate_size=8960,
                initializer_range=0.02,
                max_position_embeddings=131072,
                model_type="qwen2",
                max_window_layers=28,
                num_attention_heads=12,
                num_hidden_layers=28,
                num_key_value_heads=2,
                rms_norm_eps=1e-06,
                rope_theta=1000000.0,
                sliding_window=131072,
                tie_word_embeddings=True,
                torch_dtype="bfloat16",
                use_cache=True,
                use_mrope=False,
                use_sliding_window=False,
            )
        elif isinstance(self.qwenexp_config, dict):
            # Override expert default config for Vanilla Qwen2
            if "model_type" not in qwenexp_config:
                qwenexp_config["model_type"] = "qwen2"

            cfg_cls = CONFIG_MAPPING[qwenexp_config["model_type"]]
            self.qwenexp_config = cfg_cls(**qwenexp_config)

        if paligemma_config is None:
            # Default config from Pi0
            self.paligemma_config = CONFIG_MAPPING["paligemma"](
                transformers_version="4.48.1",
                _vocab_size=257152,
                bos_token_id=2,
                eos_token_id=1,
                hidden_size=2048,
                image_token_index=257152,
                model_type="paligemma",
                pad_token_id=0,
                projection_dim=2048,
                text_config={
                    "hidden_activation": "gelu_pytorch_tanh",
                    "hidden_size": 2048,
                    "intermediate_size": 16384,
                    "model_type": "gemma",
                    "num_attention_heads": 8,
                    "num_hidden_layers": 18,
                    "num_image_tokens": 256,
                    "num_key_value_heads": 1,
                    "torch_dtype": "float32",
                    "vocab_size": 257152,
                },
                vision_config={
                    "hidden_size": 1152,
                    "intermediate_size": 4304,
                    "model_type": "siglip_vision_model",
                    "num_attention_heads": 16,
                    "num_hidden_layers": 27,
                    "num_image_tokens": 256,
                    "patch_size": 14,
                    "projection_dim": 2048,
                    "projector_hidden_act": "gelu_fast",
                    "torch_dtype": "float32",
                    "vision_use_head": False,
                },
            )
        elif isinstance(self.paligemma_config, dict):
            # Override Pi0 default config for PaliGemma
            if "model_type" not in gemma_expert_config:
                paligemma_config["model_type"] = "paligemma"

            cfg_cls = CONFIG_MAPPING[paligemma_config["model_type"]]
            self.paligemma_config = cfg_cls(**paligemma_config)

        if gemma_expert_config is None:
            # Default config from Pi0
            self.gemma_expert_config = CONFIG_MAPPING["gemma"](
                attention_bias=False,
                attention_dropout=0.0,
                bos_token_id=2,
                eos_token_id=1,
                head_dim=256,
                hidden_act="gelu_pytorch_tanh",
                hidden_activation="gelu_pytorch_tanh",
                hidden_size=1024,
                # hidden_size = 2048,
                initializer_range=0.02,
                intermediate_size=4096,
                max_position_embeddings=8192,
                model_type="gemma",
                # num_attention_heads=8,
                num_attention_heads=32,
                num_hidden_layers=18,
                # num_key_value_heads=1,
                num_key_value_heads=2,
                pad_token_id=0,
                rms_norm_eps=1e-06,
                rope_theta=10000.0,
                torch_dtype="float32",
                transformers_version="4.48.1",
                use_cache=True,
                vocab_size=257152,
            )
        elif isinstance(self.gemma_expert_config, dict):
            # Override Pi0 default config for Gemma Expert
            if "model_type" not in gemma_expert_config:
                gemma_expert_config["model_type"] = "gemma"

            cfg_cls = CONFIG_MAPPING[paligemma_config["model_type"]]
            self.gemma_expert_config = cfg_cls(**gemma_expert_config)

        super().__init__(**kwargs)

    def __post_init__(self):
        super().__post_init__()
        if self.train_expert_only and not self.freeze_vision_encoder:
            raise ValueError(
                "You set `freeze_vision_encoder=False` and `train_expert_only=True` which are not compatible."
            )

        if self.attention_implementation not in ["eager", "fa2", "flex"]:
            raise ValueError(
                f"Wrong value provided for `attention_implementation` ({self.attention_implementation}). Expected 'eager', 'fa2' or 'flex'."
            )
            
class DimensionalExpansion(nn.Module):
    def __init__(self, in_dim=8, out_dim=64):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)  # 或 nn.BatchNorm1d(out_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        # 输入形状: [batch, seq_len, in_dim, hidden] -> [4, 868, 8, 256]
        batch, seq_len, in_dim, hidden = x.shape
        x = x.permute(0, 1, 3, 2)  # 交换维度 -> [4, 868, 256, 8]
        x = x.reshape(-1, in_dim)   # 合并维度 -> [4*868*256, 8]

        # 线性变换扩展特征维度
        x = self.linear(x)          # -> [4*868*256, 64]
        x = self.norm(x)            # 归一化层
        x = x.view(batch, seq_len, hidden, -1)  # 恢复形状 -> [4, 868, 256, 64]
        x = x.permute(0, 1, 3, 2)  # 调整维度顺序 -> [4, 868, 64, 256]

        # 激活函数
        x = self.activation(x)
        return x

class DimensionalSqueezeBack(nn.Module):
    def __init__(self, in_dim=16384, out_dim=2048):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.activation = nn.GELU()
    
    def forward(self, x):
        # 输入形状: [batch, seq_len, hidden] -> [4, 868, 16384]
        batch, seq_len, hidden = x.shape
        x = x.reshape(-1, hidden)  # 合并维度 -> [4*868, 16384]

        # 线性变换压缩特征维度
        x = self.linear(x)          # -> [4*868, 2048]
        x = self.norm(x)            # 归一化层
        x = x.view(batch, seq_len, -1)  # 恢复形状 -> [4, 868, 2048]
        
        # 激活函数
        x = self.activation(x)

        return x

class PaliGemmaWithExpertModel(PreTrainedModel):
    config_class = PaliGemmaWithExpertConfig

    def __init__(self, config: PaliGemmaWithExpertConfig):
        super().__init__(config=config)
        self.config = config
        self.qwen25vl = Qwen2_5_VLForConditionalGeneration(config=config.qwen25vl_config)
        self.qwen_expert = Qwen2ForCausalLM(config=config.qwenexp_config)
        
        # Remove unused embed_tokens
        self.qwen_expert.model.embed_tokens = None
        
        # num_layers = self.paligemma.config.text_config.num_hidden_layers
        # self.query_expansion_layers = nn.ModuleList([
        #     DimensionalExpansion(in_dim=8, out_dim=32) for _ in range(num_layers)
        # ])
        # self.key_expansion_layers = nn.ModuleList([
        #     DimensionalExpansion(in_dim=1, out_dim=2) for _ in range(num_layers)
        # ])
        
        # self.value_expansion_layers = nn.ModuleList([
        #     DimensionalExpansion(in_dim=1, out_dim=2) for _ in range(num_layers)
        # ])
        # self.attn_compression_layers = nn.ModuleList([
        #     DimensionalSqueezeBack(in_dim=2048*4, out_dim=2048) for _ in range(num_layers)
        # ])

        self.to_bfloat16_like_physical_intelligence()
        self.set_requires_grad()
        
    def init_load(self, path):
        self.qwen25vl = Qwen2_5_VLForConditionalGeneration.from_pretrained(path)

    def set_requires_grad(self):
        if self.config.freeze_vision_encoder:
            self.paligemma.vision_tower.eval()
            for params in self.paligemma.vision_tower.parameters():
                params.requires_grad = False

        if self.config.train_expert_only:
            self.paligemma.eval()
            for params in self.paligemma.parameters():
                params.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)

        if self.config.freeze_vision_encoder:
            self.paligemma.vision_tower.eval()

        if self.config.train_expert_only:
            self.paligemma.eval()

    def to_bfloat16_like_physical_intelligence(self):
        self.qwen25vl = self.qwen25vl.to(dtype=torch.bfloat16)
        self.qwen_expert = self.qwen_expert.to(dtype=torch.bfloat16)
        # self.paligemma = self.paligemma.to(dtype=torch.bfloat16)

        params_to_change_dtype = [
            "language_model.model.layers",
            "qwen_expert.model.layers",
            "visual",
            # "multi_modal",
        ]
        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_change_dtype):
                param.data = param.data.to(dtype=torch.bfloat16)

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.model.embed_tokens(tokens)

    # TODO: break down this huge forward into modules or functions
    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        inputs_embeds: List[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        fill_kv_cache: Optional[bool] = None,
    ):
        models = [self.qwen25vl.model, self.qwen_expert.model]
        
        outputs_embeds = []

        for i, hidden_states in enumerate(inputs_embeds):
            # TODO this is very inefficient
            # dtype is always the same, batch size too (if > 1 len)
            # device could be trickier in multi gpu edge cases but that's it
            if hidden_states is None:
                continue
            batch_size = hidden_states.shape[0]
            if i == 0:
                outputs = models[i].forward(
                    input_ids = None,
                    position_ids = position_ids,
                    attention_mask = attention_mask,
                    past_key_values = past_key_values,
                    inputs_embeds = hidden_states,
                    use_cache = use_cache,
                    output_hidden_states=True
                )
                outputs_embeds.append(outputs.hidden_states[-1])
                if use_cache and past_key_values is None:
                    past_key_values = outputs.past_key_values
            elif i == 1:
                outputs = models[i].forward(
                    input_ids = None,
                    position_ids = None,
                    attention_mask = attention_mask,
                    past_key_values = past_key_values,
                    inputs_embeds = hidden_states,
                    use_cache = use_cache,
                    output_hidden_states=True
                )
                outputs_embeds.append(outputs.hidden_states[-1])
                
          
        return outputs_embeds, past_key_values

    def get_attention_interface(self):
        if self.config.attention_implementation == "fa2":
            attention_interface = self.flash_attention_forward
        elif self.config.attention_implementation == "flex":
            attention_interface = flex_attention_forward
        else:
            attention_interface = self.eager_attention_forward
        return attention_interface

    def flash_attention_forward(
        self, attention_mask, batch_size, head_dim, query_states, key_states, value_states
    ):
        raise NotImplementedError("FA2 is not implemented (yet)")

    def eager_attention_forward(
        self, attention_mask, batch_size, head_dim, query_states, key_states, value_states
    ):
        # gemma_expert_config
        num_att_heads = self.config.gemma_expert_config.num_attention_heads
        num_key_value_heads = self.config.gemma_expert_config.num_key_value_heads
        num_key_value_groups = num_att_heads // num_key_value_heads
        # # paligemma_config
        # num_att_heads = self.config.paligemma_config.text_config.num_attention_heads
        # num_key_value_heads = self.config.paligemma_config.text_config.num_key_value_heads
        # num_key_value_groups = num_att_heads // num_key_value_heads

        # query_states: batch_size, sequence_length, num_att_head, head_dim
        # key_states: batch_size, sequence_length, num_key_value_head, head_dim
        # value_states: batch_size, sequence_length, num_key_value_head, head_dim
        sequence_length = key_states.shape[1]

        key_states = key_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        key_states = key_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        value_states = value_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        value_states = value_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        # Attention here is upcasted to float32 to match the original eager implementation.

        query_states = query_states.to(dtype=torch.float32)
        key_states = key_states.to(dtype=torch.float32)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        att_weights *= head_dim**-0.5
        big_neg = -2.3819763e38  # See gemma/modules.py

        masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)

        probs = nn.functional.softmax(masked_att_weights, dim=-1)
        probs = probs.to(dtype=value_states.dtype)

        # probs: batch_size, num_key_value_head, num_att_head, sequence_length, sequence_length
        # value_states: batch_size, sequence_length, num_att_heads, head_dim

        att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))

        att_output = att_output.permute(0, 2, 1, 3)
        # we use -1 because sequence length can change
        att_output = att_output.reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)

        return att_output