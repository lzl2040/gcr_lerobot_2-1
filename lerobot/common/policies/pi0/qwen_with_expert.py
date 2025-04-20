from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.version
from pytest import Cache
from torch import nn
import transformers
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
import transformers.modeling_outputs
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.auto import CONFIG_MAPPING
from transformers.cache_utils import DynamicCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from torch.utils.checkpoint import checkpoint

from lerobot.common.policies.pi0.flex_attention import flex_attention_forward

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

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

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class PaliGemmaWithExpertConfig(PretrainedConfig):
    model_type = "PaliGemmaWithExpertModel"
    sub_configs = {"qwen25vl_config": AutoConfig, "qwenexp_config": AutoConfig}

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
                attn_implementation = "flash_attention_2",
            )
        elif isinstance(self.qwenexp_config, dict):
            # Override expert default config for Vanilla Qwen2
            if "model_type" not in qwenexp_config:
                qwenexp_config["model_type"] = "qwen2"

            cfg_cls = CONFIG_MAPPING[qwenexp_config["model_type"]]
            self.qwenexp_config = cfg_cls(**qwenexp_config)

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
    
class KVCompress(nn.Module):
    def __init__(self, in_dim=4, out_dim=2):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim
    
    def forward(self, t):
        # 输入形状: [batch, num_kv_heads, seq_len, head_dim] -> [B, 4, S, 128]
        new_t = []
        for x in t:
            batch, num_kv_heads, seq_len, head_dim = x.shape
            assert num_kv_heads == self.in_dim, f"num_kv_heads must be equal to in_dim, which is {self.in_dim}, but got {num_kv_heads}."
            x = x.reshape(-1, num_kv_heads)  # 合并维度 -> [128*B*S, 4]

            # 线性变换压缩特征维度
            x = self.linear(x)          # -> [4*868, 2048]
            
            x = x.view(batch, self.out_dim, seq_len, head_dim)  # 恢复形状 -> [4, 868, 2048]
            new_t.append(x)
        
        # 激活函数
        # x = self.activation(x)

        return new_t

class PaliGemmaWithExpertModel(PreTrainedModel):
    config_class = PaliGemmaWithExpertConfig

    def __init__(self, config: PaliGemmaWithExpertConfig, init_load = False, init_path = None):
        super().__init__(config=config)
        self.config = config
        self.gradient_checkpointing = True
        # print(config.qwen25vl_config)
        # print(config.qwenexp_config)
        config.qwenexp_config._attn_implementation_internal = "flash_attention_2"
        config.qwen25vl_config._attn_implementation_internal = "flash_attention_2"
        if not init_load:
            self.qwen25vl = Qwen2_5_VLForConditionalGeneration(config=config.qwen25vl_config)
        else:
            self.qwen25vl = Qwen2_5_VLForConditionalGeneration.from_pretrained(init_path)
        self.qwen_expert = Qwen2ForCausalLM(config=config.qwenexp_config)
        
        self.kv_compress = KVCompress(in_dim=4, out_dim=2)
        
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
            print("Freezing vision encoder")
            self.qwen25vl.visual.eval()
            for params in self.qwen25vl.visual.parameters():
                params.requires_grad = False

        if self.config.train_expert_only:
            print("Freezing qwen25vl")
            self.qwen25vl.eval()
            for params in self.qwen25vl.parameters():
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
                    
                    outputs.past_key_values.key_cache = self.kv_compress(outputs.past_key_values.key_cache)
                    outputs.past_key_values.value_cache = self.kv_compress(outputs.past_key_values.value_cache)
                    past_key_values = outputs.past_key_values
                    # print(f"past_key_values: {past_key_values.key_cache[0].shape}")
            elif i == 1:
                # print(f"attention_mask: {attention_mask.shape}, {attention_mask[:, -1].sum().item()}")
                # print(f"input tensor :{hidden_states.size()},  {hidden_states.size()[0]}")
                outputs = self.expert_forward(
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

    def expert_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
        ):
        output_attentions = output_attentions if output_attentions is not None else self.qwen_expert.model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.qwen_expert.model.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.qwen_expert.model.config.use_cache
        return_dict = return_dict if return_dict is not None else self.qwen_expert.model.config.use_return_dict
        
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            # print(
            #     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            # )
            use_cache = False
        
        if inputs_embeds is None:
            inputs_embeds = self.qwen_expert.model.embed_tokens(input_ids)
            
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
            
        causal_mask = self.qwen_expert.model._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        
        hidden_states = inputs_embeds
        position_embeddings = self.qwen_expert.model.rotary_emb(hidden_states, position_ids)
        
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        
        for layer_idx in range(self.qwen_expert.model.config.num_hidden_layers):
            decoder_layer = self.qwen_expert.model.layers[layer_idx]
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                
            if self.gradient_checkpointing and self.training:
                if layer_idx % 14 == 0:
                    layer_outputs = checkpoint(
                        self.custom_decoder_forward,
                        decoder_layer,
                        hidden_states,
                        causal_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        cache_position,
                        position_embeddings,
                        use_reentrant=False,
                    )
                else:
                    layer_outputs = self.custom_decoder_forward(
                        decoder_layer,
                        hidden_states,
                        causal_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        cache_position,
                        position_embeddings
                    )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )
            
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        
        hidden_states = self.qwen_expert.model.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        output = transformers.modeling_outputs.BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        
        return output if return_dict else output.to_tuple()
        
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
    
    def custom_decoder_forward(
        self,
        decoder_layer,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        residual = hidden_states
        
        hidden_states = decoder_layer.input_layernorm(hidden_states)
        
        hidden_states, self_attn_weights = self.custom_attention_forward(
            decoder_layer.self_attn,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        
        # Fully Connected
        residual = hidden_states
        hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
        hidden_states = decoder_layer.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs
    
    def custom_attention_forward(
        self,
        attn_layer,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn_layer.head_dim)
        
        query_states = attn_layer.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = attn_layer.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = attn_layer.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        if past_key_value is not None:
            key_cache, value_cache = past_key_value[attn_layer.layer_idx]
            if key_cache is not None:
                key_states = torch.cat([key_cache, key_states], dim=-2)
            if value_cache is not None:
                value_states = torch.cat([value_cache, value_states], dim=-2)
                
        sliding_window = None
        if (
            attn_layer.config.use_sliding_window
            and getattr(attn_layer.config, "sliding_window", None) is not None
            and attn_layer.layer_idx >= attn_layer.config.max_window_layers
        ):
            sliding_window = attn_layer.config.sliding_window
            
        attention_interface: Callable = eager_attention_forward
        if attn_layer.config._attn_implementation != "eager":
            if attn_layer.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                print(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[attn_layer.config._attn_implementation]
                
        attn_output, attn_weights = attention_interface(
            attn_layer,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not attn_layer.training else attn_layer.attention_dropout,
            scaling=attn_layer.scaling,
            sliding_window=sliding_window,  # main diff with Llama
            **kwargs,
        )
        
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_layer.o_proj(attn_output)
        
        return attn_output, attn_weights
    
    def custom_visual_forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor):
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        # print(f"Into custom visual processing, input states: \nhidden_states: {hidden_states.shape}, grid_thw: {grid_thw.shape}")
        hidden_states = self.qwen25vl.visual.patch_embed(hidden_states)
        # print(f"After patch embedding, hidden_states: {hidden_states.shape}")
        rot_pos_emb = self.qwen25vl.visual.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.qwen25vl.visual.get_window_index(grid_thw)
        
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        seq_len, _ = hidden_states.size()
        # print(f"Sequence length: {seq_len}, cu_window_seqlens: {cu_window_seqlens.shape}")
        hidden_states = hidden_states.reshape(seq_len // self.qwen25vl.visual.spatial_merge_unit, self.qwen25vl.visual.spatial_merge_unit, -1)
        # print(f"After first reshaping, hidden_states: {hidden_states.shape}")
        
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        
        rot_pos_emb = rot_pos_emb.reshape(seq_len // self.qwen25vl.visual.spatial_merge_unit, self.qwen25vl.visual.spatial_merge_unit, -1)
        rot_pos_emb = rot_pos_emb[window_index, :, :]
        rot_pos_emb = rot_pos_emb.reshape(seq_len, -1)
        
        emb = torch.cat((rot_pos_emb, rot_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)
        
        for layer_num, blk in enumerate(self.qwen25vl.visual.blocks):
            if layer_num in self.qwen25vl.visual.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings)
            
        hidden_states = self.qwen25vl.visual.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]
        
        return hidden_states
        