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
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLAttention
import transformers.modeling_outputs
from transformers.modeling_flash_attention_utils import _flash_attention_forward
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
    if q is not None:
        q_embed = (q * cos) + (rotate_half(q) * sin)
    else:
        q_embed = None
    if k is not None:
        k_embed = (k * cos) + (rotate_half(k) * sin)
    else:
        k_embed = None
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

def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension separately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
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
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

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
    sub_configs = {"qwen25vl_config": AutoConfig, "qwenexp_config": AutoConfig, "awa_model_config": AutoConfig}

    def __init__(
        self,
        paligemma_config: dict | None = None,
        gemma_expert_config: dict | None = None,
        freeze_vision_encoder: bool = True,
        train_expert_only: bool = True,
        attention_implementation: str = "eager",
        train_main_layers: int = 0,
        qwen25vl_config: dict | None = None,
        awa_model_config: dict | None = None,
        qwenexp_config: dict | None = None,
        topk: int = 8,
        **kwargs,
    ):
        self.train_main_layers = train_main_layers
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.attention_implementation = attention_implementation
        self.topk = topk
        
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
                hidden_size=1280,
                attention_dropout=0.0,
                hidden_act="silu",
                intermediate_size=7168,
                initializer_range=0.02,
                max_position_embeddings=131072,
                model_type="qwen2",
                max_window_layers=28,
                num_attention_heads=10,
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
        
        if awa_model_config is None:
            # Default config for awa model
            self.awa_model_config = CONFIG_MAPPING["qwen2"](
                transformers_version = "4.40.1",
                vocab_size=151936,
                bos_token_id=151643,
                eos_token_id=151643,
                hidden_size=1024,
                attention_dropout=0.0,
                hidden_act="silu",
                intermediate_size=5376,
                initializer_range=0.02,
                max_position_embeddings=131072,
                model_type="qwen2",
                max_window_layers=28,
                num_attention_heads=8,
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
        elif isinstance(self.awa_model_config, dict):
            # Override expert default config for Vanilla Qwen2
            if "model_type" not in awa_model_config:
                awa_model_config["model_type"] = "qwen2"

            cfg_cls = CONFIG_MAPPING[awa_model_config["model_type"]]
            self.awa_model_config = cfg_cls(**qwenexp_config)

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
    
class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
    
class KVCompress(nn.Module):
    def __init__(self, in_dim=4, out_dim=2, hiddem_dim=128):
        super().__init__()
        self.expand = nn.Linear(in_dim*hiddem_dim, in_dim*hiddem_dim)
        self.linear = nn.Linear(in_dim*hiddem_dim, out_dim*hiddem_dim)
        self.norm = Qwen2RMSNorm(hidden_size=in_dim*hiddem_dim)
        self.activation = nn.SiLU()
        self.in_dim = in_dim
        self.out_dim = out_dim
    
    def forward(self, t):
        # 输入形状: [batch, num_kv_heads, seq_len, head_dim] -> [B, 4, S, 128]
        new_t = []
        for x in t:
            batch, num_kv_heads, seq_len, head_dim = x.shape
            assert num_kv_heads == self.in_dim, f"num_kv_heads must be equal to in_dim, which is {self.in_dim}, but got {num_kv_heads}."
            x = x.reshape(-1, num_kv_heads*head_dim)  # 合并维度 -> [B*S, 4*128]

            # 短路缓存
            y = x.clone()
            
            # 线性变换特征维度
            x = self.expand(x)          # -> [B*S, 2*128]
            x = self.norm(x)
            
            x = self.activation(x)
            x = x + y
            
            x = self.linear(x)          # -> [B*S, 2*128]
            
            x = x.view(batch, self.out_dim, seq_len, head_dim)  # 恢复形状 -> [4, 868, 2048]
            new_t.append(x)

        return new_t
    
class QueryCompression(nn.Module):
    def __init__(self, in_dim=2, out_dim=4, hiddem_dim=128):
        super().__init__()
        self.linear = nn.Linear(out_dim*hiddem_dim, out_dim*hiddem_dim)
        self.norm = Qwen2RMSNorm(hidden_size=out_dim*hiddem_dim)
        self.activation = nn.SiLU()
        self.compress = nn.Linear(in_dim*hiddem_dim, out_dim*hiddem_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x):
        
        x_shape = x.shape # B L N H
        x = x.reshape(x_shape[0], x_shape[1], self.in_dim*x_shape[-1]) # B L N*H
        
        x = self.compress(x)          # -> [B*S, 2*128]
        x = self.norm(x)
        x = self.activation(x)
        x = self.linear(x)          # -> [B*S, 2*128]
        
        x = x.reshape(x_shape[0], x_shape[1], self.out_dim, x_shape[-1])  # B L N* H
        
        return x
        
    
class KvRepresentation(nn.Module):
    def __init__(self, hidden=128, in_head=[6, 4], out_head=2):
        super().__init__()
        # self.linear = nn.Linear(hidden*in_head, hidden*in_head)
        # self.norm = Qwen2RMSNorm(hidden*in_head)
        # self.activate_1 = nn.SiLU()
        # self.linear1 = nn.Linear(hidden*in_head[0]*in_head[1], hidden*in_head[0]*in_head[1])
        # self.norm_1 = Qwen2RMSNorm(hidden*in_head[0]*in_head[1])
        
        # self.kv_mask = nn.Parameter(torch.ones(hidden*in_head[0]*in_head[1]*2))
        
        self.compress_to_tgtdim = nn.Linear(hidden*in_head[0]*in_head[1], hidden*out_head)
        self.linear_2 = nn.Linear(hidden*out_head, hidden*out_head)
        self.norm_3 = Qwen2RMSNorm(hidden*out_head)
        self.activate_3 = nn.SiLU()
        
        self.in_head = in_head
        self.out_head = out_head
        
    def forward(self, x):
        # x -> (batch_size, seq_len, head1*head2, head_dim)
        
        x = x.reshape(x.shape[0], -1, x.shape[2]*x.shape[3]) # -> (batch_size, seq_len, head1*head2*head_dim)
        # y = x.clone()
        # x = self.linear1(x) # -> [B, len, 128, 128]
        # x = self.norm_1(x)
        # x = self.activate_1(x)
        # x = x + y
        
        x = self.compress_to_tgtdim(x) # -> (batch_size, seq_len, head2*head_dim)
        # x = x.transpose(-1, -2) # -> [B, len, 2, 128]
        
        y = x.clone()
        
        x = self.norm_3(x)
        x = self.activate_3(x)
        x = self.linear_2(x) # -> (batch_size, seq_len, head2*head_dim)
        
        x = x + y
        
        x = x.view(x.shape[0], x.shape[1], self.out_head, -1) # -> (batch_size, seq_len, head_dim, out_head)
        
        return x
        

class PaliGemmaWithExpertModel(PreTrainedModel):
    config_class = PaliGemmaWithExpertConfig

    def __init__(self, config: PaliGemmaWithExpertConfig, init_load = False, init_path = None):
        super().__init__(config=config)
        self.cross_forward_flag = True
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
        # self.awa_model = Qwen2ForCausalLM(config=config.awa_model_config)
        
        self.num_layers = self.config.qwen25vl_config.num_hidden_layers
        # self.kv_compress = nn.ModuleList([KVCompress(in_dim=4, out_dim=2) for _ in range(self.num_layers)])
        self.kv_repre = nn.ModuleList([KvRepresentation(hidden=128, in_head=[8, 4], out_head=2) for _ in range(self.num_layers)])
        
        self.k_mask = [nn.Parameter(torch.ones(128*8*4)) for _ in range(self.num_layers)]
        self.v_mask = [nn.Parameter(torch.ones(128*8*4)) for _ in range(self.num_layers)]
        
        # self.query_compress = nn.ModuleList([QueryCompression(in_dim=6, out_dim=4, hiddem_dim=128) for _ in range(self.num_layers)])
        # self.kv_compress = KVCompress(in_dim=4, out_dim=2)
        self.awa_model = Qwen2ForCausalLM(config=config.awa_model_config)
        del self.awa_model.lm_head
        del self.qwen_expert.lm_head
        
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
        if self.config.train_expert_only:
            print(f"Freezing qwen25vl, setting {self.config.train_main_layers} layers unfrozen")
            if self.config.train_main_layers == 0:
                self.qwen25vl.eval()
                for params in self.qwen25vl.parameters():
                    params.requires_grad = False
            else:
                self.qwen25vl.model.train()
                for params in self.qwen25vl.parameters():
                    params.requires_grad = False
                for layer_idx in range(self.config.train_main_layers):
                    for params in self.qwen25vl.model.layers[-layer_idx-1].parameters():
                        params.requires_grad = True
        else:
            print("Training qwen25vl")
            self.qwen25vl.train()
            for params in self.qwen25vl.parameters():
                params.requires_grad = True
                        
        if self.config.freeze_vision_encoder:
            print("Freezing vision encoder")
            self.qwen25vl.visual.eval()
            for params in self.qwen25vl.visual.parameters():
                params.requires_grad = False
        else:
            print("Training vision encoder")
            self.qwen25vl.visual.train()
            for params in self.qwen25vl.visual.parameters():
                params.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)

        if self.config.freeze_vision_encoder:
            self.qwen25vl.visual.eval()

        # if self.config.train_expert_only:
        #     self.paligemma.eval()

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
        if not self.cross_forward_flag:
            models = [self.qwen25vl.model, self.awa_model, self.qwen_expert.model]
            
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
                        use_cache = True,
                        output_hidden_states=True
                    )
                    outputs_embeds.append(outputs.hidden_states[-1])
                    if use_cache and past_key_values is None:
                        
                        # for layer_idx in range(self.num_layers):
                        #     outputs.past_key_values.key_cache[layer_idx] = self.kv_compress[layer_idx](outputs.past_key_values.key_cache[layer_idx])[0]
                        #     outputs.past_key_values.value_cache[layer_idx] = self.kv_compress[layer_idx](outputs.past_key_values.value_cache[layer_idx])[0]
                        # outputs.past_key_values.key_cache = self.kv_compress(outputs.past_key_values.key_cache)
                        # outputs.past_key_values.value_cache = self.kv_compress(outputs.past_key_values.value_cache)
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
        else:
            hidden_state_vl, hidden_state_awa, hidden_state_exp, kv_holder =self.cross_forward(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                fill_kv_cache=fill_kv_cache
            )
            return [hidden_state_vl, hidden_state_awa, hidden_state_exp], kv_holder
    
    def cross_forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        inputs_embeds: List[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        fill_kv_cache: Optional[bool] = None):
        use_cache = use_cache if use_cache is not None else False
        output_attentions, output_hidden_states, use_cache, return_dict, position_ids_vl, cache_position_vl, causal_mask_vl, position_embeddings_vl, cache_position_awa,  position_ids_awa, causal_mask_awa, position_embeddings_awa, cache_position_exp, position_ids_exp, causal_mask_exp, position_embeddings_exp = self.custom_set_inputs(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds
        )
        
        models = [self.qwen25vl.model, self.awa_model.model, self.qwen_expert.model]
        
        hidden_states = inputs_embeds
                
        num_layers = self.num_layers
        for layer_idx in range(num_layers):
            if layer_idx % 1 == 0:
                hidden_states = checkpoint(
                    self.cross_layer_forward,
                    models,
                    layer_idx,
                    hidden_states,
                    attention_mask,
                    causal_mask_vl,
                    causal_mask_awa,
                    causal_mask_exp,
                    position_ids_vl,
                    position_ids_awa,
                    position_ids_exp,
                    output_attentions,
                    cache_position_vl,
                    cache_position_awa,
                    cache_position_exp,
                    position_embeddings_vl,
                    position_embeddings_awa,
                    position_embeddings_exp,
                    use_reentrant=False,
                )
            else:
                hidden_states = self.cross_layer_forward(models,
                                                        layer_idx,
                                                        inputs_embeds=hidden_states,
                                                        attention_mask=attention_mask,
                                                        casual_mask_vl=causal_mask_vl,
                                                        casual_mask_awa=causal_mask_awa,
                                                        casual_mask_exp=causal_mask_exp,
                                                        position_ids_vl=position_ids_vl,
                                                        position_ids_awa=position_ids_awa,
                                                        position_ids_exp=position_ids_exp,
                                                        output_attentions=output_attentions,
                                                        cache_position_vl=cache_position_vl,
                                                        cache_position_awa=cache_position_awa,
                                                        cache_position_exp=cache_position_exp,
                                                        position_embeddings_vl=position_embeddings_vl,
                                                        position_embeddings_awa=position_embeddings_awa,
                                                        position_embeddings_exp=position_embeddings_exp
                                                        )
        hidden_state_vl, hidden_state_awa, hidden_state_exp = hidden_states
        hidden_state_exp = self.qwen_expert.model.norm(hidden_state_exp)
        
        return hidden_state_vl, hidden_state_awa, hidden_state_exp, None
    
    def custom_set_inputs(
        self, 
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        inputs_embeds: Optional[List[torch.FloatTensor]] = None,
        ):
        output_attentions = self.qwen25vl.config.output_attentions
        output_hidden_states = (True)
        use_cache = False
        return_dict = True
        inputs_embeds_vl = inputs_embeds[0]
        if (inputs_embeds_vl is None):
            raise ValueError("You must specify inputs_embeds of QwenVL model")
        past_seen_tokens = 0
        cache_position_vl = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds_vl.shape[1], device=inputs_embeds_vl.device
            )
        if position_ids is None:
            position_ids = cache_position_vl.view(1, 1, -1).expand(3, inputs_embeds_vl.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        causal_mask_vl = self.qwen25vl.model._update_causal_mask(
            attention_mask, inputs_embeds_vl, cache_position_vl, past_key_values, output_attentions
        )
        
        # create position embeddings to be shared across the decoder layers
        position_embeddings_vl = self.qwen25vl.model.rotary_emb(inputs_embeds_vl, position_ids)
        
        sample_key_values = DynamicCache()
        seq_len = inputs_embeds_vl.shape[1]
        bsize = inputs_embeds_vl.shape[0]
        hidden_dim = inputs_embeds_vl.shape[2]
        head_dim = hidden_dim // self.qwen25vl.config.num_attention_heads
        num_kv_heads = self.qwen_expert.config.num_key_value_heads
        sample_key = torch.randn([bsize, num_kv_heads, seq_len, head_dim], device=inputs_embeds_vl.device)
        sample_value = torch.randn([bsize, num_kv_heads, seq_len, head_dim], device=inputs_embeds_vl.device)
        sample_key, sample_value = sample_key_values.update(sample_key, sample_value, 0)
        
        # prepare inputs for awa model
        inputs_embeds_awa = inputs_embeds[1]
        past_seen_tokens = sample_key_values.get_seq_length()
        cache_position_awa = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds_awa.shape[1], device=inputs_embeds_awa.device
            )
        position_ids_awa = cache_position_awa.unsqueeze(0)
        causal_mask_awa = self.awa_model.model._update_causal_mask(
                attention_mask, inputs_embeds_awa, cache_position_awa, sample_key_values, output_attentions
            )
        position_embeddings_awa = self.awa_model.model.rotary_emb(inputs_embeds_awa, position_ids_awa)
        
        # prepare inputs for expert model
        inputs_embeds_exp = inputs_embeds[2]
        past_seen_tokens += 1
        cache_position_exp = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds_exp.shape[1], device=inputs_embeds_exp.device
            )
        position_ids_exp = cache_position_exp.unsqueeze(0)
        causal_mask_exp = self.qwen_expert.model._update_causal_mask(
                attention_mask, inputs_embeds_exp, cache_position_exp, sample_key_values, output_attentions
        )
        position_embeddings_exp = self.qwen_expert.model.rotary_emb(inputs_embeds_exp, position_ids_exp)
        
        
        return output_attentions, output_hidden_states, use_cache, return_dict, position_ids, cache_position_vl, causal_mask_vl, position_embeddings_vl, cache_position_awa, position_ids_awa, causal_mask_awa, position_embeddings_awa, cache_position_exp, position_ids_exp, causal_mask_exp, position_embeddings_exp
    
    def cross_layer_forward(
        self, 
        models, 
        layer_idx, 
        inputs_embeds, 
        attention_mask,
        casual_mask_vl, 
        casual_mask_awa,
        casual_mask_exp, 
        position_ids_vl,
        position_ids_awa,
        position_ids_exp, 
        output_attentions, 
        cache_position_vl,
        cache_position_awa,
        cache_position_exp, 
        position_embeddings_vl,
        position_embeddings_awa,
        position_embeddings_exp,
        ):
        
        layers = [model.layers[layer_idx] for model in models]
        
        if inputs_embeds[0] is not None:
            hidden_states_vl = inputs_embeds[0]
            residual_vl = hidden_states_vl
            
            # print(f"RMSNorm param at layer {layer_idx} is: {layers[0].input_layernorm.weight.size()}")
            hidden_states_vl = layers[0].input_layernorm(hidden_states_vl)
            
            hidden_states_vl, self_attn_weights, present_key_value = self.qwen_vl_flow_attn(
                attn=layers[0].self_attn,
                hidden_states=hidden_states_vl,
                position_ids=position_ids_vl,
                past_key_value=None,
                attention_mask=casual_mask_vl,
                output_attentions=output_attentions,
                use_cache=False,
                cache_position=cache_position_vl,
                position_embeddings=position_embeddings_vl
            )
            
            # if isinstance(present_key_value, list):
            #     present_key_value = self.kv_compress[layer_idx](present_key_value)
            # else:
            #     present_key_value.key_cache = self.kv_compress[layer_idx](present_key_value.key_cache)
            #     present_key_value.value_cache = self.kv_compress[layer_idx](present_key_value.value_cache)
                
            if inputs_embeds[1] is not None:
                inputs_embeds_awa = inputs_embeds[1]
                
                awa_outputs = self.awa_decoder_forward(
                    decoder_layer=layers[1],
                    layer_idx=layer_idx,
                    hidden_states=inputs_embeds_awa,
                    attention_mask=casual_mask_awa,
                    position_ids=position_ids_awa,
                    past_key_value=present_key_value,
                    output_attentions=output_attentions,
                    output_kv=True,
                    use_cache=False,
                    cache_position=cache_position_awa,
                    position_embeddings=position_embeddings_awa
                )
                awa_hidden = awa_outputs[0]
                present_key_value = awa_outputs[2] if output_attentions else awa_outputs[1]
                
            
            if inputs_embeds[2] is not None:
                expert_hidden_states = inputs_embeds[2]
                
                expert_outputs = self.expert_decoder_forward(
                    decoder_layer=layers[2],
                    hidden_states=expert_hidden_states,
                    attention_mask=casual_mask_exp,
                    position_ids=position_ids_exp,
                    past_key_value=present_key_value,
                    output_attentions=output_attentions,
                    output_kv=False,
                    use_cache=False,
                    cache_position=cache_position_exp,
                    position_embeddings=position_embeddings_exp
                )
                expert_hidden = expert_outputs[0]
           
            hidden_states_vl = residual_vl + hidden_states_vl
            
            residual_vl = hidden_states_vl
            hidden_states_vl = layers[0].post_attention_layernorm(hidden_states_vl)
            hidden_states_vl = layers[0].mlp(hidden_states_vl)
            hidden_states_vl = residual_vl + hidden_states_vl
        
        outputs = [hidden_states_vl, awa_hidden, expert_hidden]
        return outputs

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
            # Use cache not ppssible in gradient checkpointing mode
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
                        self.expert_decoder_forward,
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
                    layer_outputs = self.expert_decoder_forward(
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
        
    def awa_decoder_forward(
        self,
        decoder_layer,
        layer_idx: int,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        output_kv: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        assert past_key_value is not None 
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, decoder_layer.self_attn.head_dim)
        
        state_query = decoder_layer.self_attn.q_proj(hidden_states).view(hidden_shape) # (batch_size, seq_len, num_heads, head_dim)
        
        # state_query = self.query_compress[layer_idx](state_query)
        
        cos, sin = position_embeddings
        state_query, _ = apply_rotary_pos_emb(state_query, None, cos, sin)
        # repeat state_query until it reach length of kv 
        seq_len = past_key_value[0].shape[2]
        state_query = state_query.expand(-1, seq_len, -1, -1)
        # state_query = state_query[:, :, :, :, None] # (batch_size, num_heads, seq_len, head_dim)
        # key = past_key_value[0].unsqueeze(-1).transpose(-1, -2).transpose(2, 1) 
        # (batch_size, num_heads, seq_len, head_dim) -> (batch_size, num_heads, seq_len, 1, head_dim) -> (batch_size, seq_len, num_heads, 1, head_dim)
        # value = past_key_value[1].unsqueeze(-1).transpose(-1, -2).transpose(2, 1)  
        # (batch_size, num_heads, seq_len, head_dim) -> (batch_size, num_heads, seq_len, 1, head_dim) -> (batch_size, seq_len, num_heads, 1, head_dim)
        key = past_key_value[0].transpose(2, 1)
        # (batch_size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, num_heads, head_dim)
        value = past_key_value[1].transpose(2, 1)
        # (batch_size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, num_heads, head_dim)
        
        head_dim = key.shape[-1]
        scaling_factor = torch.sqrt(torch.tensor(head_dim, dtype=key.dtype, device=key.device))
        key_weight_awa = torch.einsum("b s a d, b s c d -> b s a c d", state_query, key) / scaling_factor # (batch_size, seq_len, head_dim, head_dim)
        value_weight_awa = torch.einsum("b s a d, b s c d -> b s a c d", state_query, value) / scaling_factor # (batch_size, seq_len, head_dim, head_dim)
        key_weight_awa_shape = key_weight_awa.shape # (batch_size, seq_len, head_dim, head_dim)
        key_weight_awa = key_weight_awa.view(key_weight_awa_shape[0], key_weight_awa_shape[1], -1, key_weight_awa_shape[4])
        value_weight_awa = value_weight_awa.view(key_weight_awa_shape[0], key_weight_awa_shape[1], -1, key_weight_awa_shape[4])
        
        key_weight_awa = key_weight_awa.view(key_weight_awa_shape[0], key_weight_awa_shape[1], -1)
        value_weight_awa = value_weight_awa.view(key_weight_awa_shape[0], key_weight_awa_shape[1], -1)
        key_weight_awa = self.k_mask[layer_idx].to(device=key_weight_awa.device)*key_weight_awa
        value_weight_awa = self.v_mask[layer_idx].to(device=value_weight_awa.device)*value_weight_awa
        key_weight_awa = key_weight_awa.view(key_weight_awa_shape[0], key_weight_awa_shape[1], -1, key_weight_awa_shape[4])
        value_weight_awa = value_weight_awa.view(key_weight_awa_shape[0], key_weight_awa_shape[1], -1, key_weight_awa_shape[4])
        
        # weight_awa_threshold, weight_awa_index = torch.topk(key_weight_awa, k=self.config.topk, dim=2, sorted=False)
        # filter_mask = torch.zeros_like(key_weight_awa, dtype=torch.bool, device=key_weight_awa.device)
        # filter_mask.scatter_(2, weight_awa_index, True)
        # key_weight_awa = key_weight_awa * filter_mask.to(key_weight_awa.dtype)
        # value_weight_awa = value_weight_awa * filter_mask.to(key_weight_awa.dtype)
        # (batch_size, seq_len, head1*head2, head_dim)
        
        key_awa = self.kv_repre[layer_idx](key_weight_awa).transpose(2, 1)
        value_awa = self.kv_repre[layer_idx](value_weight_awa).transpose(2, 1)
        # (batch_size, seq_len, num_heads, head_dim)
        
        past_key_value = [key_awa, value_awa]
        
        residual = hidden_states
        
        hidden_states = decoder_layer.input_layernorm(hidden_states)
        
        hidden_states, self_attn_weights, key_value = self.expert_attention_forward(
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
        if output_kv:
            outputs += (key_value,)

        return outputs
    
    def expert_decoder_forward(
        self,
        decoder_layer,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        output_kv: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        residual = hidden_states
        
        hidden_states = decoder_layer.input_layernorm(hidden_states)
        
        hidden_states, self_attn_weights, key_value = self.expert_attention_forward(
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
        if output_kv:
            outputs += (key_value,)

        return outputs
    
    def qwen_vl_flow_attn(
        self,
        attn,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        ):
        bsz, q_len, _ = hidden_states.size()
        
        # print(f"hidden shape: {hidden_states.size()}")
        # print(f"attn info: {attn.q_proj.weight.size()}, {attn.q_proj.bias.size()}")
        query_states = attn.q_proj(hidden_states)
        key_states = attn.k_proj(hidden_states)
        value_states = attn.v_proj(hidden_states)
        
        query_states = query_states.view(bsz, q_len, -1, attn.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, attn.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, attn.head_dim).transpose(1, 2)

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, attn.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        else:
            past_key_value = [key_states, value_states]

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, attn.num_key_value_groups)
        value_states = repeat_kv(value_states, attn.num_key_value_groups)
        # dropout_rate = 0.0 if not attn.training else attn.attention_dropout

        causal_mask = attention_mask
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        
        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
        
        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        is_causal = True if causal_mask is None and q_len > 1 else False
        
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=attn.attention_dropout if attn.training else 0.0,
            is_causal=is_causal)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, attn.hidden_size)
        
        attn_output = attn.o_proj(attn_output)
        
        return attn_output, None, past_key_value
    
    def expert_attention_forward(
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
            if isinstance(past_key_value, list):
                key_cache, value_cache = past_key_value
            else:
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
        
        return attn_output, attn_weights, [key_states, value_states]
    
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
        