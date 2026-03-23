#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import os

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LLAMA_INPUTS_DOCSTRING
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
# from transformers.utils import add_start_docstrings_to_model_forward
from .llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
import logging
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

logger = logging.getLogger(__name__)

# 尝试导入Flash Attention
FLASH_ATTN_AVAILABLE = False
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
    logger.info("Flash Attention is available and can be used with FLASHUSE=1")
except ImportError:
    logger.warning("Flash Attention is not available. Install with: pip install flash-attn")
    flash_attn_func = None

def add_start_docstrings_to_model_forward(*docstr):
    def docstring_decorator(fn):
        docstring = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        class_name = f"[`{fn.__qualname__.split('.')[0]}`]"
        intro = f"   The {class_name} forward method, overrides the `__call__` special method."
        note = r"""

    <Tip>

    Although the recipe for forward pass needs to be defined within this function, one should call the [`Module`]
    instance afterwards instead of this since the former takes care of running the pre and post processing steps while
    the latter silently ignores them.

    </Tip>
"""

        fn.__doc__ = intro + note + docstring
        return fn

    return docstring_decorator

class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"
    _attn_implementation = "sdpa"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)
    
    def forward_with_flash_attention(
        self,
        inputs_embeds: torch.FloatTensor,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        使用Flash Attention的前向传播（仅支持简单mask场景）
        
        Args:
            inputs_embeds: (B, L, D) 输入嵌入
            position_ids: (B, L) 位置ID
            attention_mask: (B, L) 2D padding mask，True表示有效位置
            is_causal: 是否使用因果mask
            use_cache: 不支持，必须为False
            output_attentions: 不支持，必须为False
            output_hidden_states: 是否输出所有hidden states
            return_dict: 是否返回字典
        
        Returns:
            BaseModelOutputWithPast
        """
        if not FLASH_ATTN_AVAILABLE:
            raise RuntimeError("Flash Attention is not available. Please install flash-attn.")
        
        if use_cache:
            raise NotImplementedError("Flash Attention path does not support use_cache=True")
        
        if output_attentions:
            raise NotImplementedError("Flash Attention does not return attention weights")
        
        batch_size, seq_length, _ = inputs_embeds.shape
        hidden_states = inputs_embeds
        
        # 应用RoPE需要的cos/sin（从第一个attention层获取）
        # Flash Attention内部会处理RoPE，但我们需要确保position_ids正确
        
        all_hidden_states = () if output_hidden_states else None
        
        # 处理padding mask：Flash Attention使用不同的格式
        # attention_mask: (B, L) bool tensor, True表示有效位置
        # Flash Attention需要知道每个样本的实际长度或使用key_padding_mask
        key_padding_mask = None
        if attention_mask is not None:
            # 将bool mask转换为Flash Attention的格式
            # Flash Attention: True表示需要mask掉，False表示保留
            # 我们的mask: True表示有效，False表示padding
            # 所以需要取反
            key_padding_mask = ~attention_mask.bool()  # (B, L), True表示padding位置
        
        # 遍历每一层
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            # 获取当前层的attention和mlp模块
            self_attn = decoder_layer.self_attn
            
            # === 手动实现Attention with Flash Attention ===
            residual = hidden_states
            hidden_states = decoder_layer.input_layernorm(hidden_states)
            
            # Q, K, V投影
            bsz, q_len, _ = hidden_states.size()
            query_states = self_attn.q_proj(hidden_states)
            key_states = self_attn.k_proj(hidden_states)
            value_states = self_attn.v_proj(hidden_states)
            
            # Reshape: (B, L, num_heads, head_dim)
            query_states = query_states.view(bsz, q_len, self_attn.num_heads, self_attn.head_dim)
            key_states = key_states.view(bsz, q_len, self_attn.num_key_value_heads, self_attn.head_dim)
            value_states = value_states.view(bsz, q_len, self_attn.num_key_value_heads, self_attn.head_dim)
            
            # 应用RoPE
            kv_seq_len = key_states.shape[1]
            # 转换为(B, num_heads, L, head_dim)进行RoPE
            query_states_rope = query_states.transpose(1, 2)
            key_states_rope = key_states.transpose(1, 2)
            value_states_rope = value_states.transpose(1, 2)
            
            cos, sin = self_attn.rotary_emb(value_states_rope, seq_len=kv_seq_len)
            
            query_states_rope, key_states_rope = apply_rotary_pos_emb(
                query_states_rope, key_states_rope, cos, sin, position_ids
            )
            
            # 转回(B, L, num_heads, head_dim)
            query_states = query_states_rope.transpose(1, 2)
            key_states = key_states_rope.transpose(1, 2)
            value_states = value_states_rope.transpose(1, 2)
            
            # 处理GQA：如果num_key_value_heads < num_heads，需要repeat
            if self_attn.num_key_value_groups > 1:
                # repeat_kv: (B, L, num_kv_heads, head_dim) -> (B, L, num_heads, head_dim)
                key_states = key_states.repeat_interleave(self_attn.num_key_value_groups, dim=2)
                value_states = value_states.repeat_interleave(self_attn.num_key_value_groups, dim=2)
            
            # Flash Attention调用
            # flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, 
            #                 window_size=(-1, -1), alibi_slopes=None, deterministic=False)
            # 输入格式：(batch_size, seqlen, nheads, headdim)
            
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout_p=0.0,
                softmax_scale=None,  # 默认使用1/sqrt(head_dim)
                causal=is_causal,
                # key_padding_mask 参数在flash-attn 2.3.2中可能不直接支持
                # 需要使用cu_seqlens或直接忽略padding（如果都是满的）
            )
            
            # 如果有padding mask，手动处理padding位置的输出
            if key_padding_mask is not None:
                # 将padding位置的输出置零
                attn_output = attn_output.masked_fill(
                    key_padding_mask.unsqueeze(-1).unsqueeze(-1),  # (B, L, 1, 1)
                    0.0
                )
            
            # Reshape回(B, L, hidden_size)
            attn_output = attn_output.reshape(bsz, q_len, self_attn.num_heads * self_attn.head_dim)
            
            # Output投影
            attn_output = self_attn.o_proj(attn_output)
            attn_output = torch.nan_to_num(attn_output, nan=0.0, posinf=1e5, neginf=-1e5)
            
            # Residual连接
            hidden_states = residual + attn_output
            
            # === MLP部分（保持不变）===
            residual = hidden_states
            hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
            hidden_states = decoder_layer.mlp(hidden_states)
            hidden_states = residual + hidden_states
            hidden_states = torch.nan_to_num(hidden_states, nan=0.0, posinf=1e5, neginf=-1e5)
        
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        if not return_dict:
            return tuple(v for v in [hidden_states, None, all_hidden_states, None] if v is not None)
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=None,
        )

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward_support_4d(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # ----------- 原版前半段保持不变 -----------
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        seq_length_with_past, past_key_values_length = seq_length, 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past += past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            ).unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # ----------- 这里开始改动：统一生成 `decoder_attention_mask` -----------
        if attention_mask is None:
            # 与原版一致：默认全 1
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )

        if attention_mask.dim() == 4:
            # 用户传入的已扩展 4-D mask:
            #   ― 如果是 bool：True 表示可见 -> 转成 0 / -inf additive mask
            #   ― 如果已经是 float/half：默认视为 additive mask，直接使用
            if attention_mask.dtype == torch.bool:
                attn_dtype = inputs_embeds.dtype
                attention_mask = attention_mask.to(attn_dtype)
                # True → 0.0   False → -inf
                attention_mask = (1.0 - attention_mask) * torch.finfo(attn_dtype).min
            # 保证 device 和 dtype
            attention_mask = attention_mask.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            combined_attention_mask = attention_mask  # 已经是最终形状
        else:
            # 维持原版行为（2-D bool → causal + padding mask）
            combined_attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # 后续逻辑 **完全照搬** 原实现，只把 `attention_mask` 换成 `combined_attention_mask`
        hidden_states = inputs_embeds
        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`")
            use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_self_attns  = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                def custom_forward(*inputs):
                    return decoder_layer(*inputs, output_attentions, None)
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    custom_forward,
                    hidden_states,
                    combined_attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=combined_attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]
            hidden_states = torch.nan_to_num(hidden_states, nan=0.0, posinf=1e5, neginf=-1e5)

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state = hidden_states,
            past_key_values   = next_cache,
            hidden_states     = all_hidden_states,
            attentions        = all_self_attns,
        )

class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.hidden_size = config.hidden_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.pretraining_tp = config.pretraining_tp

        if self.config.add_dist_token:
            number_tokens = list(range(32000,32256))
            weighted_mask = torch.ones(32256)
            self.config.vocab_size = 32256
        else:
            number_tokens = [
                    718,
                    448,
                    29900,
                    29889,
                    29896,
                    29906,
                    29941,
                    29946,
                    29945,
                    29953,
                    29955,
                    29947,
                    29929,
                ]  # +-0.123456789
            weighted_mask = torch.ones(self.config.vocab_size)
            number_tokens += [28544, 
                              11255, 
                              1563, 
                              1266, 
                              4151, 
                              523, 
                              3396, 
                              2408]
            # slow, fast, left, right, 'stra', 'ight', 'main', 'tain'
        weighted_mask[number_tokens] = 3.0
        self.register_buffer("weighted_mask", weighted_mask)
        
        # Initialize weights and apply final processing
        self.post_init()


        # self.query_norm = nn.LayerNorm(self.hidden_size)

    def get_model(self):
        return self.model

    # def switch_lora_adapter(self, adapter_name: str):
    #     """
    #     切换当前激活的LoRA adapter（双LoRA模式下使用）
        
    #     Args:
    #         adapter_name: adapter名称，如 "lora_1" 或 "lora_2"
    #     """
    #     if hasattr(self, 'set_adapter'):
    #         self.set_adapter(adapter_name)
    #         print(f"Switched to LoRA adapter: {adapter_name}")
    #     else:
    #         print("Warning: Model does not have multiple adapters")
    
    # def enable_lora_adapters(self, adapter_names: list):
    #     """
    #     启用多个LoRA adapters（可以同时使用两个LoRA）
        
    #     Args:
    #         adapter_names: adapter名称列表，如 ["lora_1", "lora_2"]
    #     """
    #     if hasattr(self, 'set_adapter'):
    #         self.set_adapter(adapter_names)
    #         print(f"Enabled LoRA adapters: {adapter_names}")
    #     else:
    #         print("Warning: Model does not have multiple adapters")
    
    def get_active_adapters(self):
        """
        获取当前激活的LoRA adapters
        
        Returns:
            当前激活的adapter名称或列表
        """
        if hasattr(self, 'active_adapters'):
            return self.active_adapters
        else:
            return None

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        points: Optional[torch.FloatTensor] = None,
        trajectories: Optional[torch.FloatTensor] = None,
        ego_features: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            if trajectories is not None or points is not None:
                (
                    input_ids, 
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds, # 序列长度与下列的labels长度相同
                    labels # 新的labels序列会变长，主要是将图像特征的序列和其他特征的序列对应的部分也进行IGNORE
                ) = self.prepare_inputs_labels_for_multimodal_traj(
                    input_ids,
                    position_ids, # None
                    attention_mask,
                    past_key_values,
                    labels,
                    images,
                    points,
                    trajectories,
                    ego_features, # None
                    image_sizes,
       
                )   

            else:
                (
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    images,
                    image_sizes
                )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(weight=self.weighted_mask.float())
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            loss = torch.nan_to_num(loss)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
       
# 在 llava_llama.py 的 LlavaLlamaForCausalLM 类里添加：

    def sanitize_tensor(self, x: torch.Tensor, eps: float = 1e-5):
        # 复用你已有的 sanitize 逻辑，顺便提供一个小工具
        if torch.isnan(x).any():
            print("Warning: NaNs detected. Replacing with zeros.")
            x = torch.nan_to_num(x, nan=0.0)
        if torch.isinf(x).any():
            print("Warning: Infs detected. Clipping to finite range.")
            x = torch.clamp(x, min=-1e4, max=1e4)
        max_abs = torch.max(torch.abs(x))
        if max_abs > 1e4:
            print(f"Warning: Very large values (max={max_abs.item():.2e}). Clipping.")
            x = torch.clamp(x, min=-1e4, max=1e4)
        return x


    def forward_queries_train(
        self,
        base_inputs_embeds: torch.Tensor,           # [B, T, D]  (文本/多模态已在此)
        query_embeds: torch.Tensor,                 # [B, Q, D]  (可学习 queries)
        attention_mask: torch.Tensor = None,        # [B, T] 或 [B, T+Q]，其中“第二答+padding=False”
        position_ids: torch.LongTensor = None,      # [B, T] 或 [B, T+Q]
        past_key_values: list = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        detach: bool = False,
        return_full_hidden: bool = False,
        use_text_context: bool = False,
        text_embs_stage: str = "initial", # initial middle last
    ):
        """
        训练版：
        - 假定 attention_mask 已将“第二个回答 + padding”置为 False
        - queries 仅可看到 (Q1 + A1 + Q2)，看不到第二答与 padding
        - base 看不到 queries；base 内部保持因果
        - 第二答与 padding 在注意力中既不能作为 key 被看见，也不能作为 query 去看别人
        返回:
        query_feats: [B, Q, D] 最后一层隐藏态
        updated: dict，包含拼接后的 2D/4D 掩码与位置等
        (可选) all_hidden_states
        """

        # ---- 形状检查 ----
        B, T, D = base_inputs_embeds.shape
        Bq, Q, Dq = query_embeds.shape
        assert B == Bq, "Batch size mismatch."
        assert D == Dq == self.model.config.hidden_size, \
            f"Hidden size mismatch: base={D}, query={Dq}, model={self.model.config.hidden_size}"

        device = base_inputs_embeds.device
        llm = self.model
        llm_param = next(llm.parameters())
        llm_dtype  = llm_param.dtype
        llm_device = llm_param.device
        NEG_INF = torch.finfo(llm_dtype).min
        POS_INF = torch.finfo(llm_dtype).max

        # ---- 数值清洗（与你的测试版一致）----
        base_inputs_embeds = sanitize_inputs_embeds(base_inputs_embeds)
        query_embeds = self.sanitize_tensor(query_embeds)

        # ---- 拼接 ----
        inputs_embeds = torch.cat([base_inputs_embeds, query_embeds], dim=1)  # [B, T+Q, D]
        L = T + Q

        # ---- attention_mask 补齐到 [B, T+Q]，queries 默认有效(True) ----
        if attention_mask is None:
            attention_mask = torch.ones(B, T, dtype=torch.long, device=device)
        assert attention_mask.dim() == 2 and attention_mask.size(0) == B
        if attention_mask.size(1) == T:
            q_mask = torch.ones(B, Q, dtype=attention_mask.dtype, device=device)
            attention_mask = torch.cat([attention_mask, q_mask], dim=1)       # [B, T+Q]
        else:
            assert attention_mask.size(1) == L, "Unexpected attention_mask length."

        # ---- position_ids 补齐到 [B, T+Q] ----
        if position_ids is None:
            position_ids = torch.arange(0, L, dtype=torch.long, device=device).unsqueeze(0).repeat(B, 1)
        else:
            assert position_ids.dim() == 2 and position_ids.size(0) == B
            if position_ids.size(1) == T:
                last_pos = position_ids[:, -1:]  # [B,1]
                incr = torch.arange(1, Q + 1, device=device, dtype=position_ids.dtype).unsqueeze(0)  # [1,Q]
                position_ids = torch.cat([position_ids, last_pos + incr], dim=1)                     # [B,T+Q]
            else:
                assert position_ids.size(1) == L, "Unexpected position_ids length."

        L = T + Q

        # ---- 1) 构造块状加性偏置（base 无因果；只禁止 base→query）----
        # base↔base：允许（双向，不做因果）
        base_base = torch.zeros((T, T), device=llm_device, dtype=llm_dtype)

        # base→query：禁止
        base_to_query = torch.full((T, Q), fill_value=NEG_INF, device=llm_device, dtype=llm_dtype)

        # query→base：允许
        query_to_base = torch.zeros((Q, T), device=llm_device, dtype=llm_dtype)

        # query↔query：允许（双向）
        # 若想让 query 内部因果，可改为上三角屏蔽：torch.triu(torch.ones((Q,Q), ...), diagonal=1)
        query_query = torch.zeros((Q, Q), device=llm_device, dtype=llm_dtype)

        upper = torch.cat([base_base,   base_to_query], dim=1)  # [T, L]
        lower = torch.cat([query_to_base, query_query], dim=1)  # [Q, L]
        attn_bias = torch.cat([upper, lower], dim=0)            # [L, L]

        # 扩展到 4D: [B, 1, L, L]（覆写用的“加性偏置”）
        attn_bias = attn_bias.unsqueeze(0).unsqueeze(1).expand(B, 1, L, L).contiguous()

        # ---- 2) 融合 2D padding mask（覆写而非累加）----
        mask_2d_bool = attention_mask.to(device=llm_device, dtype=torch.bool)  # [B, L]

        # 屏蔽列（key 不可被任何人看见）
        attn_bias = attn_bias.masked_fill((~mask_2d_bool).view(B, 1, 1, L), NEG_INF)

        # 屏蔽行（这些位置自己也不能去看别人）
        attn_bias = attn_bias.masked_fill((~mask_2d_bool).view(B, 1, L, 1), NEG_INF)

        # ---- 3) “全行被屏蔽”保险：给这类行放一个自环(对角=0，其余=NEG_INF) 以避免 softmax NaN ----
        # 说明：当某一行全部是 NEG_INF，softmax 会出 NaN。对 padding 行放行自环即可（不影响有效行）。
        row_all_masked = (attn_bias <= (NEG_INF * 0.5)).all(dim=-1, keepdim=True)  # [B,1,L,1]

        eye = torch.eye(L, device=llm_device, dtype=llm_dtype)                     # [L,L]
        eye = eye.unsqueeze(0).unsqueeze(1).expand(B, 1, L, L)                      # [B,1,L,L]

        fix_row = torch.full_like(attn_bias, NEG_INF)                               # 其他列仍为 NEG_INF
        fix_row = torch.where(eye.bool(), torch.zeros_like(attn_bias), fix_row)     # 仅对角置 0

        attn_bias = torch.where(row_all_masked, fix_row, attn_bias)
        # ---- 4) 双保险：移除 NaN / ±Inf（理论上不会出现，这里只是兜底）----
        attn_bias = torch.nan_to_num(attn_bias, nan=0.0, posinf=POS_INF, neginf=NEG_INF)

        # ---- dtype / device 对齐 ----
        inputs_embeds = inputs_embeds.to(device=llm_device, dtype=llm_dtype).contiguous()
        position_ids  = position_ids.to(device=llm_device, dtype=torch.long)

        # ---- 过模型（使用 4D 加性掩码）----
        outputs = self.model.forward_support_4d(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attn_bias,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )

        last_hidden = outputs.last_hidden_state           # [B, T+Q, D]
        last_hidden = torch.nan_to_num(last_hidden, nan=0.0, posinf=POS_INF, neginf=NEG_INF)

        query_feats = last_hidden[:, -Q:, :]              # [B, Q, D]

        if detach:
            query_feats = query_feats.detach()

        # 便于后续继续用
        updated = {
            "inputs_embeds": inputs_embeds,
            "attention_mask_2d": attention_mask.to(llm_device, dtype=torch.long),  # [B,L]
            "attention_mask_4d": attn_bias,                                        # [B,1,L,L]
            "position_ids": position_ids,
            "past_key_values": None,  # 训练时未启用
            "last_hidden": last_hidden
            # "hidden_states": outputs.hidden_states,
        }
        import os, ipdb
        if os.getenv("HIDDEN_DEBUG") == "1": ipdb.set_trace()  
        if return_full_hidden:
            return query_feats, updated, outputs.hidden_states
        else:
            return query_feats, updated

    def forward_queries_test(
        self,
        base_inputs_embeds: torch.Tensor,          # [B, T, D] 已有的 eLLM（图像/导航/文字）
        query_embeds: torch.Tensor,                # [B, Q, D] 外部提供的可学习 queries
        attention_mask: Optional[torch.Tensor] = None,  # [B, T] 或 [B, T+Q]
        position_ids: Optional[torch.LongTensor] = None, # [B, T] 或 [B, T+Q]
        past_key_values: Optional[list] = None,     # 可选，与常规 forward 对齐
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        detach: bool = False,                       # 如果你只想拿特征不反传，置 True
        return_full_hidden: bool = False,           # 如需调试整段 hidden，可打开
        use_text_context: bool = False,
        text_embs_stage: str = "initial", # initial middle last
    ):
        """
        将外部 learnable queries 直接拼到序列末尾，跑一遍 LLM，
        并返回这些 query 位置对应的隐藏特征，用于类外 MLP 生成轨迹。
        不计算 logits、不参与语言 loss。

        返回:
          query_feats: [B, Q, D]  —— 对应每个 query token 的最后层 hidden
          updated: dict 包含 concat 后的 inputs_embeds/attention_mask/position_ids（如你想继续生成）
          (可选) all_hidden_states: 仅 return_full_hidden=True 时返回
        """
        # import os, ipdb
        # if os.getenv("DEBUG") == "1": ipdb.set_trace()
        # 基本检查
        B, T, D = base_inputs_embeds.shape
        Bq, Q, Dq = query_embeds.shape
        assert B == Bq, "Batch size mismatch between base_inputs_embeds and query_embeds"
        assert D == Dq == self.model.config.hidden_size, \
            f"Hidden size mismatch: got D={D}, query D={Dq}, model={self.model.config.hidden_size}"

        device = base_inputs_embeds.device

        # 处理数值稳定性（可选）
        base_inputs_embeds = sanitize_inputs_embeds(base_inputs_embeds) # 也出现了NaN，已解决
        query_embeds = self.sanitize_tensor(query_embeds) # 出现了NaN，已解决

        # 拼接 embeds
        inputs_embeds = torch.cat([base_inputs_embeds, query_embeds], dim=1)  # [B, T+Q, D]
        # inputs_embeds = self.query_norm(inputs_embeds.float()).to(dtype=base_inputs_embeds.dtype)
        

        # attention mask
        if attention_mask is None: # None
            attention_mask = torch.ones(B, T, dtype=torch.long, device=device)
        if attention_mask.size(1) != T:
            # 若外部已带上 Q 段，也允许；否则我们自己补
            assert attention_mask.size(1) in (T, T+Q), "Unexpected attention_mask length"
        if attention_mask.size(1) == T:
            q_mask = torch.ones(B, Q, dtype=attention_mask.dtype, device=device)
            attention_mask = torch.cat([attention_mask, q_mask], dim=1)       # [B, T+Q]

        # position ids
        if position_ids is None:
            # 连续位置编码：0..T+Q-1
            position_ids = torch.arange(0, T + Q, dtype=torch.long, device=device).unsqueeze(0).repeat(B, 1)
        else:
            assert position_ids.size(1) in (T, T+Q), "Unexpected position_ids length"
            if position_ids.size(1) == T:
                # 延续已有位置，再为 Q 段补上增量位置
                last_pos = position_ids[:, -1:]  # [B,1]
                incr = torch.arange(1, Q + 1, device=device, dtype=position_ids.dtype).unsqueeze(0)  # [1,Q]
                pos_q = last_pos + incr
                position_ids = torch.cat([position_ids, pos_q], dim=1)  # [B, T+Q]


        llm = self.model
        llm_param = next(llm.parameters())
        llm_dtype  = llm_param.dtype
        llm_device = llm_param.device
        NEG_INF = torch.finfo(llm_dtype).min
        POS_INF = torch.finfo(llm_dtype).max

        L = T + Q

        # ---- 1) 构造块状加性偏置（base 无因果；只禁止 base→query）----
        # base↔base：允许（双向，不做因果）
        base_base = torch.zeros((T, T), device=llm_device, dtype=llm_dtype)

        # base→query：禁止
        base_to_query = torch.full((T, Q), fill_value=NEG_INF, device=llm_device, dtype=llm_dtype)

        # query→base：允许
        query_to_base = torch.zeros((Q, T), device=llm_device, dtype=llm_dtype)

        # query↔query：允许（双向）
        # 若想让 query 内部因果，可改为上三角屏蔽：torch.triu(torch.ones((Q,Q), ...), diagonal=1)
        query_query = torch.zeros((Q, Q), device=llm_device, dtype=llm_dtype)

        upper = torch.cat([base_base,   base_to_query], dim=1)  # [T, L]
        lower = torch.cat([query_to_base, query_query], dim=1)  # [Q, L]
        attn_bias = torch.cat([upper, lower], dim=0)            # [L, L]

        # 扩展到 4D: [B, 1, L, L]（覆写用的“加性偏置”）
        attn_bias = attn_bias.unsqueeze(0).unsqueeze(1).expand(B, 1, L, L).contiguous()

        # ---- 2) 融合 2D padding mask（覆写而非累加）----
        mask_2d_bool = attention_mask.to(device=llm_device, dtype=torch.bool)  # [B, L]

        # 屏蔽列（key 不可被任何人看见）
        attn_bias = attn_bias.masked_fill((~mask_2d_bool).view(B, 1, 1, L), NEG_INF)

        # 屏蔽行（这些位置自己也不能去看别人）
        attn_bias = attn_bias.masked_fill((~mask_2d_bool).view(B, 1, L, 1), NEG_INF)

        # ---- 3) “全行被屏蔽”保险：给这类行放一个自环(对角=0，其余=NEG_INF) 以避免 softmax NaN ----
        # 说明：当某一行全部是 NEG_INF，softmax 会出 NaN。对 padding 行放行自环即可（不影响有效行）。
        row_all_masked = (attn_bias <= (NEG_INF * 0.5)).all(dim=-1, keepdim=True)  # [B,1,L,1]

        eye = torch.eye(L, device=llm_device, dtype=llm_dtype)                     # [L,L]
        eye = eye.unsqueeze(0).unsqueeze(1).expand(B, 1, L, L)                      # [B,1,L,L]

        fix_row = torch.full_like(attn_bias, NEG_INF)                               # 其他列仍为 NEG_INF
        fix_row = torch.where(eye.bool(), torch.zeros_like(attn_bias), fix_row)     # 仅对角置 0

        attn_bias = torch.where(row_all_masked, fix_row, attn_bias)

        # ---- 4) 双保险：移除 NaN / ±Inf（理论上不会出现，这里只是兜底）----
        attn_bias = torch.nan_to_num(attn_bias, nan=0.0, posinf=POS_INF, neginf=NEG_INF)


        # 注意：sanitize 可能把 dtype 变成 float32，这里统一转回 LLM dtype
        inputs_embeds = inputs_embeds.to(device=llm_device, dtype=llm_dtype).contiguous()
        
        # query_embeds 在上面已经拼进去了，如果你是先拼再转，这里无需再次处理；
        # 若你是先对 query_embeds 做了独立处理，确保它也是 llm_dtype：
        # query_embeds = query_embeds.to(device=llm_device, dtype=llm_dtype)

        # 把 padding（key 侧列）并入 4D 掩码
        if attention_mask is None or attention_mask.size(1) != L:
            # 若你已有 [B,T]，上一步已补成 [B,L]
            pass
        key_pad = (1 - attention_mask).to(dtype=llm_dtype, device=llm_device)  # 1 表示要 -inf 的列
        attn_bias = attn_bias + key_pad.view(B, 1, 1, L) * NEG_INF

        # mask & pos 的 dtype 规范：mask 用 long/bool，pos 用 long
        if attention_mask is None:
            attention_mask = torch.ones(inputs_embeds.size()[:2], device=llm_device, dtype=torch.long)
        else:
            attention_mask = attention_mask.to(device=llm_device, dtype=torch.long)

        if position_ids is None:
            B, TQ, _ = inputs_embeds.shape
            position_ids = torch.arange(0, TQ, device=llm_device, dtype=torch.long).unsqueeze(0).expand(B, -1)
        else:
            position_ids = position_ids.to(device=llm_device, dtype=torch.long)

        # --- 过模型：传 4D 浮点掩码，覆盖默认 ---
        outputs = self.model.forward_support_4d(
            input_ids=None,
            inputs_embeds=inputs_embeds.to(device=llm_device, dtype=llm_dtype),
            attention_mask=attn_bias,              # 关键：传 4D 加性掩码
            position_ids=position_ids.to(llm_device),
            past_key_values=None,                  # 建议这一步不要用 pkv（见下）
            use_cache=False,                       # 不 cache，保证一次性双向
            output_hidden_states=True,
            return_dict=True,
        )

        last_hidden = outputs.last_hidden_state                  # [B, T+Q, D]
        last_hidden = torch.nan_to_num(last_hidden, nan=0.0, posinf=POS_INF, neginf=NEG_INF)
        query_feats = last_hidden[:, -Q:, :]                     # [B, Q, D]

        if detach:
            query_feats = query_feats.detach()

        updated = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": outputs.past_key_values,
            "last_hidden": last_hidden
        }
        import os, ipdb
        if os.getenv("HIDDEN_DEBUG") == "1": ipdb.set_trace()  
        if return_full_hidden:
            return query_feats, updated, outputs.hidden_states
        else:
            return query_feats, updated
        
    def forward_queries(
        self,
        base_inputs_embeds: torch.Tensor,          # [B, T, D]
        query_embeds: torch.Tensor,                # [B, Q, D] 外部提供的可学习 queries
        attention_mask: Optional[torch.Tensor] = None,  # [B, T] 或 [B, T+Q]
        position_ids: Optional[torch.LongTensor] = None, # [B, T] 或 [B, T+Q]
        past_key_values: Optional[list] = None,     # 可选，与常规 forward 对齐
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        detach: bool = False,                       # 如果你只想拿特征不反传，置 True
        return_full_hidden: bool = False,           # 如需调试整段 hidden，可打开
        training: bool = False,
        use_text_context: bool = False,
        text_embs_stage: str = "initial", # initial middle last
    ):
        if training:
            return self.forward_queries_train(
                base_inputs_embeds, query_embeds, attention_mask, position_ids, past_key_values, use_cache, output_attentions, output_hidden_states, detach, return_full_hidden, use_text_context, text_embs_stage)
        else:
            return self.forward_queries_test(
                base_inputs_embeds, query_embeds, attention_mask, position_ids, past_key_values, use_cache, output_attentions, output_hidden_states, detach, return_full_hidden, use_text_context, text_embs_stage)
    
    def forward_queries_stage_inference(
        self,
        base_context: torch.Tensor,               # (B, Q_context, D)  视觉+导航上下文
        traj_embeds_slots: torch.Tensor,          # (B, T, D)  完整的T个槽位（只有可见位置有值）
        prev_stage_query_outs: Optional[torch.Tensor],  # (B, K_all_prev, D)  所有历史stages的query outputs（累积）
        curr_stage_query: torch.Tensor,           # (B, n_curr, D)  当前stage的queries
        last_stage_len: int = 0,                  # 最后一个stage的长度（用于掩码，确保只看Stage i-1）
        traj_valid_mask: Optional[torch.Tensor] = None,  # (B, T) bool mask，标记哪些traj位置有效
        attention_mask: Optional[torch.Tensor] = None,  # (B, Q_context) base的padding mask
        position_ids: Optional[torch.LongTensor] = None,
    ):
        """
        推理时单个stage的forward，完全对齐训练时的掩码逻辑和位置编码。
        
        序列结构（对应训练时的 [base, gt_traj_embeds, all_prev_stages_queries, curr_stage_queries]）：
            [base_context, traj_embeds_slots, prev_stage_query_outs(所有历史), curr_stage_query]
            [Q_context,    T,                 K_all_prev,                      n_curr]
        
        关键改进：
            - 传入**所有历史 stages** 的 query outputs（不只是 Stage i-1），确保位置编码与训练时对齐
            - 通过 last_stage_len 参数和掩码设计，确保 curr 只能看到 Stage i-1（最后一个 stage）
            - 这样既保证了位置编码对齐，又保证了信息流对齐
        
        掩码逻辑（完全对齐训练）：
            1. base ↔ base: 双向可见
            2. traj_slots ↔ traj_slots: 只有对角线可见（自环，避免不同时间步泄漏）
            3. base → (traj + prev + curr): 全部禁止
            4. traj → base: 可见
            5. traj → (prev + curr): 禁止
            6. prev → base: 可见
            7. prev → traj: 可见（历史已经看过）
            8. prev ↔ prev: 双向可见（但curr只能看到最后 last_stage_len 个，即 Stage i-1）
            9. prev → curr: 禁止（不能看未来）
            10. curr → base: 可见
            11. curr → traj: 可见（当前stage需要的信息）
            12. curr → prev: 只能看到最后 last_stage_len 个（Stage i-1），与训练时对齐
            13. curr ↔ curr: 双向可见
        
        Args:
            base_context: 视觉上下文 (B, Q_context, D)
            traj_embeds_slots: 完整T个嵌入槽位，只有当前可见位置有值 (B, T, D)
            prev_stage_query_outs: 所有历史stages的query outputs (B, K_all_prev, D)，首次为None
            curr_stage_query: 当前stage的queries (B, n_curr, D)
            last_stage_len: 最后一个stage（Stage i-1）的长度，用于掩码限制curr只看Stage i-1
            traj_valid_mask: 标记traj_embeds_slots中哪些位置有效 (B, T)，True表示有效
            attention_mask: base的padding mask (B, Q_context)
        
        Returns:
            curr_query_out: (B, n_curr, D)
        """
        B, Q_context, D = base_context.shape
        _, T, _ = traj_embeds_slots.shape
        _, n_curr, _ = curr_stage_query.shape
        K_prev = prev_stage_query_outs.shape[1] if prev_stage_query_outs is not None else 0
        
        # 如果没有提供 traj_valid_mask，默认所有位置有效
        if traj_valid_mask is None:
            traj_valid_mask = torch.ones(B, T, dtype=torch.bool, device=base_context.device)
        
        device = base_context.device
        llm = self.model
        llm_param = next(llm.parameters())
        llm_dtype = llm_param.dtype
        llm_device = llm_param.device
        NEG_INF = torch.finfo(llm_dtype).min
        POS_INF = torch.finfo(llm_dtype).max
        
        # ---- 数值清洗 ----
        base_context = sanitize_inputs_embeds(base_context)
        traj_embeds_slots = self.sanitize_tensor(traj_embeds_slots)
        curr_stage_query = self.sanitize_tensor(curr_stage_query)
        if prev_stage_query_outs is not None:
            prev_stage_query_outs = self.sanitize_tensor(prev_stage_query_outs)
        
        # ---- 拼接 embeds ----
        if prev_stage_query_outs is not None:
            inputs_embeds = torch.cat([base_context, traj_embeds_slots, prev_stage_query_outs, curr_stage_query], dim=1)
        else:
            inputs_embeds = torch.cat([base_context, traj_embeds_slots, curr_stage_query], dim=1)
        L = Q_context + T + K_prev + n_curr
        
        # ---- 构造掩码（完全对齐训练时的逻辑）----
        attn_mask_2d = torch.zeros((L, L), device=llm_device, dtype=llm_dtype)
        
        # 1. traj_slots ↔ traj_slots: 只有对角线可见
        traj_start = Q_context
        traj_end = Q_context + T
        traj_block = torch.full((T, T), fill_value=NEG_INF, device=llm_device, dtype=llm_dtype)
        diag_idx = torch.arange(T, device=llm_device)
        traj_block[diag_idx, diag_idx] = 0.0
        attn_mask_2d[traj_start:traj_end, traj_start:traj_end] = traj_block
        
        # 2. base → 所有后面的（traj + prev + curr）: 禁止
        attn_mask_2d[:Q_context, Q_context:] = NEG_INF
        
        # 3. traj → prev: 禁止
        if K_prev > 0:
            prev_start = Q_context + T
            prev_end = Q_context + T + K_prev
            attn_mask_2d[traj_start:traj_end, prev_start:prev_end] = NEG_INF
        
        # 4. traj → curr: 禁止
        curr_start = Q_context + T + K_prev
        attn_mask_2d[traj_start:traj_end, curr_start:] = NEG_INF
        
        # 5. prev → curr: 禁止（如果有prev）
        if K_prev > 0:
            attn_mask_2d[prev_start:prev_end, curr_start:] = NEG_INF
        
        # 6. curr → prev: 只能看到最后 last_stage_len 个（Stage i-1），与训练时对齐
        if K_prev > 0 and last_stage_len > 0 and last_stage_len < K_prev:
            # 禁止 curr 看到 prev 的前 (K_prev - last_stage_len) 个位置
            attn_mask_2d[curr_start:, prev_start:prev_start+(K_prev-last_stage_len)] = NEG_INF
        
        # 其余默认可见（已初始化为0）
        
        # ---- 扩展到 4D 并融合 padding mask ----
        attn_bias = attn_mask_2d.unsqueeze(0).unsqueeze(1).expand(B, 1, L, L).contiguous()
        
        # 处理 base 的 padding mask
        if attention_mask is None:
            attention_mask = torch.ones(B, Q_context, dtype=torch.bool, device=device)
        else:
            attention_mask = attention_mask.to(device=device, dtype=torch.bool)
        
        # traj: 使用传入的 traj_valid_mask（只有有值的位置才有效）
        # prev, curr: 默认全部有效
        traj_mask = traj_valid_mask.to(device=device, dtype=torch.bool)  # 使用传入的 mask
        prev_mask = torch.ones(B, K_prev, dtype=torch.bool, device=device) if K_prev > 0 else None
        curr_mask = torch.ones(B, n_curr, dtype=torch.bool, device=device)
        
        if prev_mask is not None:
            mask_2d_bool = torch.cat([attention_mask, traj_mask, prev_mask, curr_mask], dim=1)
        else:
            mask_2d_bool = torch.cat([attention_mask, traj_mask, curr_mask], dim=1)
        
        # 屏蔽列（key不可见）和行（query不能看）
        attn_bias = attn_bias.masked_fill((~mask_2d_bool).view(B, 1, 1, L), NEG_INF)
        attn_bias = attn_bias.masked_fill((~mask_2d_bool).view(B, 1, L, 1), NEG_INF)
        
        # "全行被屏蔽"保险（避免softmax NaN）
        row_all_masked = (attn_bias <= (NEG_INF * 0.5)).all(dim=-1, keepdim=True)
        eye = torch.eye(L, device=llm_device, dtype=llm_dtype).unsqueeze(0).unsqueeze(1).expand(B, 1, L, L)
        fix_row = torch.full_like(attn_bias, NEG_INF)
        fix_row = torch.where(eye.bool(), torch.zeros_like(attn_bias), fix_row)
        attn_bias = torch.where(row_all_masked, fix_row, attn_bias)
        
        # 数值清洗
        attn_bias = torch.nan_to_num(attn_bias, nan=0.0, posinf=POS_INF, neginf=NEG_INF)
        
        # ---- position_ids（关键：必须与训练时的绝对位置对齐）----
        # 训练时的位置：[0...Q_context-1, Q_context...Q_context+T-1, Q_context+T...Q_context+T+K_prev-1, Q_context+T+K_prev...]
        # 为了对齐，我们需要使用与训练时相同的绝对位置编码
        if position_ids is None:
            # 生成与训练时一致的绝对位置编码
            # base: 0...Q_context-1
            # traj: Q_context...Q_context+T-1  
            # prev: Q_context+T...Q_context+T+K_prev-1
            # curr: Q_context+T+K_prev...Q_context+T+K_prev+n_curr-1
            position_ids = torch.arange(0, L, dtype=torch.long, device=device).unsqueeze(0).repeat(B, 1)
        else:
            # 如果外部提供了 base 的 position_ids，需要正确扩展
            if position_ids.size(1) == Q_context:
                # 按照训练时的逻辑扩展：连续递增
                last_pos = position_ids[:, -1:]
                # traj 部分：last_pos + 1 到 last_pos + T
                incr_traj = torch.arange(1, T + 1, device=device, dtype=position_ids.dtype).unsqueeze(0)
                pos_traj = last_pos + incr_traj
                
                # prev 部分：last_pos + T + 1 到 last_pos + T + K_prev
                incr_prev = torch.arange(T + 1, T + K_prev + 1, device=device, dtype=position_ids.dtype).unsqueeze(0)
                pos_prev = last_pos + incr_prev if K_prev > 0 else None
                
                # curr 部分：last_pos + T + K_prev + 1 到 last_pos + T + K_prev + n_curr
                incr_curr = torch.arange(T + K_prev + 1, T + K_prev + n_curr + 1, device=device, dtype=position_ids.dtype).unsqueeze(0)
                pos_curr = last_pos + incr_curr
                
                # 拼接
                if pos_prev is not None:
                    position_ids = torch.cat([position_ids, pos_traj, pos_prev, pos_curr], dim=1)
                else:
                    position_ids = torch.cat([position_ids, pos_traj, pos_curr], dim=1)
        
        # ---- dtype/device 对齐 ----
        inputs_embeds = inputs_embeds.to(device=llm_device, dtype=llm_dtype).contiguous()
        position_ids = position_ids.to(device=llm_device, dtype=torch.long)
        
        # ---- 过模型（使用4D掩码）----
        outputs = self.model.forward_support_4d(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attn_bias,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        
        last_hidden = outputs.last_hidden_state  # (B, L, D)
        last_hidden = torch.nan_to_num(last_hidden, nan=0.0, posinf=POS_INF, neginf=NEG_INF)
        
        # 提取 curr_stage_query 对应的输出
        curr_query_out = last_hidden[:, -n_curr:, :]  # (B, n_curr, D)
        
        return curr_query_out

    def forward_queries_parallel(
        self,
        base_inputs_embeds: torch.Tensor,           # [B, T, D]  视觉上下文
        query_embeds: torch.Tensor,                 # [B, Q, D]  所有时间步的 queries
        stage_indices: List[List[int]],             # 多尺度索引，如 [[5], [2,4], [1,3,5], [0,1,2,3,4,5]]
        gt_traj_embeds: Optional[torch.Tensor] = None,   # (B, T, d_in)
        attention_mask: Optional[torch.Tensor] = None,  # [B, T] base 的 padding mask
        position_ids: Optional[torch.LongTensor] = None,
        training: bool = False,
        detach: bool = False,
        return_full_hidden: bool = False,
    ):
        """
        并行解码多尺度轨迹点，通过设计因果掩码确保尺度间的自回归依赖。
        
        核心思想（参考 VAR）：
        - 为每个时间步分配尺度级别（stage level）
        - 构造掩码：时间步 i 只能看到尺度级别 <= 自己的时间步 + 所有 base
        - 一次性并行解码所有点，避免 for 循环
        
        Args:
            base_inputs_embeds: (B, T, D) 视觉+导航上下文
            query_embeds: (B, Q, D) 所有时间步的 query（已加时间位置编码+条件化调制）
            stage_indices: 多尺度索引列表，如 [[5], [2,4], [1,3,5], [0,1,2,3,4,5]]
            gt_traj_embeds: (B, T, d_in) 真实轨迹点嵌入
            attention_mask: (B, T) base 的 padding mask
            training: 是否训练模式
        
        Returns:
            query_feats: (B, Q, D) 每个时间步的输出特征
            updated: dict 包含拼接后的 inputs_embeds/attention_mask 等
        """
        # ---- 形状检查 ----
        B, Q_context, D = base_inputs_embeds.shape
        Bq, T, Dq = query_embeds.shape
        assert B == Bq and D == Dq == self.model.config.hidden_size
        assert gt_traj_embeds.shape[1] == T

        device = base_inputs_embeds.device
        llm = self.model
        llm_param = next(llm.parameters())
        llm_dtype = llm_param.dtype
        llm_device = llm_param.device
        NEG_INF = torch.finfo(llm_dtype).min
        POS_INF = torch.finfo(llm_dtype).max
        
        # ---- 数值清洗 ----
        base_inputs_embeds = sanitize_inputs_embeds(base_inputs_embeds)
        query_embeds = self.sanitize_tensor(query_embeds)
        import os, ipdb
        if os.getenv("DEBUG") == "1": ipdb.set_trace() 
        # tgt reconstruction
        stage_tgt_list = []
        stage_len_list = []
        for i in range(len(stage_indices)):
            stage_tgt_list.append(query_embeds[:, stage_indices[i], :])
            stage_len_list.append(len(stage_indices[i]))
        final_tgt = torch.cat(stage_tgt_list, dim=1)
        assert final_tgt.shape[1] == sum(stage_len_list)
        Q_final = final_tgt.shape[1]

        # index list: 记录每个 stage 在 inputs_embeds 中的起始位置
        final_index_list = []
        for i in range(len(stage_len_list)):
            final_index_list.append(Q_context + T + sum(stage_len_list[:i]))
        
        # 验证索引列表的正确性
        assert len(final_index_list) == len(stage_indices), \
            f"final_index_list 长度 {len(final_index_list)} 应等于 stage_indices 长度 {len(stage_indices)}"
        
        # 验证每个 stage 的起始位置
        for i in range(len(stage_indices)):
            expected_start = Q_context + T + sum(stage_len_list[:i])
            assert final_index_list[i] == expected_start, \
                f"Stage {i} 的起始位置 {final_index_list[i]} 应为 {expected_start}"

        # ---- 2. 拼接 embeds ----
        inputs_embeds = torch.cat([base_inputs_embeds, gt_traj_embeds, final_tgt], dim=1)  # [B, Q_context+T+Q_final, D]
        L = Q_context + T + Q_final
        
        # ---- 3. 构造多尺度自回归掩码 ----
        # 初始化掩码矩阵 [L, L]，默认全部可见（0）
        attn_mask_2d = torch.zeros((L, L), device=llm_device, dtype=llm_dtype)
        
        # 3.1 base 和 GT 部分：双向可见（已经是0，无需修改）
        # base ↔ base: 0
        # GT ↔ GT: 只能自环，因为相互看会泄漏不同尺度的信息
        gt_block = torch.full((T, T), fill_value=NEG_INF, device=llm_device, dtype=llm_dtype)
        diag_idx = torch.arange(T, device=llm_device)
        gt_block[diag_idx, diag_idx] = 0.0
        attn_mask_2d[Q_context:Q_context+T, Q_context:Q_context+T] = gt_block

        # base ↔ GT: 不能看
        # 3.2 base → tgt: 禁止（base 不应看到未来的 query）
        attn_mask_2d[:Q_context, Q_context:] = NEG_INF
        
        # 3.3 GT → tgt: 禁止（GT 不应看到未来的 query）
        attn_mask_2d[Q_context:Q_context+T, Q_context+T:] = NEG_INF
        
        # 3.4 tgt → base: 允许（已经是0）
        
        # 3.5 tgt → GT: 根据 stage 规则设置
        # 默认先全部禁止，然后根据每个 stage 的规则打开对应的 GT
        tgt_start = Q_context + T
        tgt_to_gt_mask = torch.full((Q_final, T), fill_value=NEG_INF, device=llm_device, dtype=llm_dtype)
        
        # 对于每个 stage，允许其看到上一个 stage 产生的 GT 点
        tgt_offset = 0
        for stage_id in range(len(stage_indices)):
            n_queries = stage_len_list[stage_id]  # 当前 stage 的 query 数量
            
            # Stage i 可以看到 stage_indices[i-1] 对应的 GT
            if stage_id > 0:
                prev_stage_gt_indices = stage_indices[stage_id - 1]
                # 将这些 GT 位置设为可见
                for gt_idx in prev_stage_gt_indices:
                    tgt_to_gt_mask[tgt_offset:tgt_offset+n_queries, gt_idx] = 0.0 # 没问题 torch.Size([21, 6])
            # else: Stage 0 不看任何 GT（保持 NEG_INF）
            
            tgt_offset += n_queries
        
        # 将 tgt → GT 的掩码填入主掩码矩阵
        attn_mask_2d[tgt_start:, Q_context:Q_context+T] = tgt_to_gt_mask
        
        # 3.6 tgt ↔ tgt: 因果掩码（Stage i 只能看到 Stage i-1 和 Stage i，不看更早的）
        # 这样设计的好处：
        # 1. 简化自回归逻辑（Stage i 基于 Stage i-1 的输出细化）
        # 2. 减少长程依赖（避免 Stage 5 直接 attend 到 Stage 0）
        # 3. 更容易与测试时对齐（测试时只需传入上一个 stage 的 outputs）
        tgt_tgt_mask = torch.full((Q_final, Q_final), fill_value=NEG_INF, device=llm_device, dtype=llm_dtype)
        
        row_offset = 0
        for i in range(len(stage_indices)):
            n_rows = stage_len_list[i]  # Stage i 的 query 数量
            
            if i == 0:
                # Stage 0: 只能看到自己
                visible_start = 0
                visible_end = stage_len_list[0]
            else:
                # Stage i (i>0): 可以看到 Stage i-1 和自己
                visible_start = sum(stage_len_list[:i-1])  # Stage i-1 的起始位置
                visible_end = sum(stage_len_list[:i+1])     # Stage i 的结束位置
            
            # 设置可见区域为 0（其他区域保持 NEG_INF）
            tgt_tgt_mask[row_offset:row_offset+n_rows, visible_start:visible_end] = 0.0
            
            row_offset += n_rows
        
        attn_mask_2d[tgt_start:, tgt_start:] = tgt_tgt_mask
        
        # ---- 4. 扩展到 4D 并融合 padding mask ----
        attn_bias = attn_mask_2d.unsqueeze(0).unsqueeze(1).expand(B, 1, L, L).contiguous()
        
        # 处理 base 的 padding mask
        if attention_mask is None:
            attention_mask = torch.ones(B, Q_context, dtype=torch.bool, device=device)
        else:
            attention_mask = attention_mask.to(device=device, dtype=torch.bool)
        
        # GT 和 query 部分默认全部有效
        gt_mask = torch.ones(B, T, dtype=torch.bool, device=device)
        q_mask = torch.ones(B, Q_final, dtype=torch.bool, device=device)
        mask_2d_bool = torch.cat([attention_mask, gt_mask, q_mask], dim=1)  # (B, L)
        
        # 屏蔽列（key 不可见）和行（query 不能看）
        attn_bias = attn_bias.masked_fill((~mask_2d_bool).view(B, 1, 1, L), NEG_INF)
        attn_bias = attn_bias.masked_fill((~mask_2d_bool).view(B, 1, L, 1), NEG_INF)
        
        # ---- 5. "全行被屏蔽"保险（避免 softmax NaN）----
        row_all_masked = (attn_bias <= (NEG_INF * 0.5)).all(dim=-1, keepdim=True)
        eye = torch.eye(L, device=llm_device, dtype=llm_dtype).unsqueeze(0).unsqueeze(1).expand(B, 1, L, L)
        fix_row = torch.full_like(attn_bias, NEG_INF)
        fix_row = torch.where(eye.bool(), torch.zeros_like(attn_bias), fix_row)
        attn_bias = torch.where(row_all_masked, fix_row, attn_bias)
        
        # ---- 6. 数值清洗 ----
        attn_bias = torch.nan_to_num(attn_bias, nan=0.0, posinf=POS_INF, neginf=NEG_INF)
        
        # ---- 7. position_ids ----
        if position_ids is None:
            position_ids = torch.arange(0, L, dtype=torch.long, device=device).unsqueeze(0).repeat(B, 1)
        else:
            if position_ids.size(1) == Q_context:
                last_pos = position_ids[:, -1:]
                incr_gt = torch.arange(1, T + 1, device=device, dtype=position_ids.dtype).unsqueeze(0)
                incr_q = torch.arange(T + 1, T + Q_final + 1, device=device, dtype=position_ids.dtype).unsqueeze(0)
                position_ids = torch.cat([position_ids, last_pos + incr_gt, last_pos + incr_q], dim=1)
        
        # ---- 8. dtype/device 对齐 ----
        inputs_embeds = inputs_embeds.to(device=llm_device, dtype=llm_dtype).contiguous()
        position_ids = position_ids.to(device=llm_device, dtype=torch.long)
        
        # ---- 9. 过模型（使用 4D 掩码）----
        outputs = self.model.forward_support_4d(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attn_bias,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            output_hidden_states=True if return_full_hidden else False,
            return_dict=True,
        )
        
        last_hidden = outputs.last_hidden_state  # [B, L, D]
        last_hidden = torch.nan_to_num(last_hidden, nan=0.0, posinf=POS_INF, neginf=NEG_INF)
        
        # 提取 final_tgt 对应的输出：[B, Q_final, D]
        final_tgt_output = last_hidden[:, tgt_start:, :]  # [B, Q_final, D]
        
        if detach:
            final_tgt_output = final_tgt_output.detach()
        
        # 返回 final_tgt_output（按 stage 顺序）和辅助信息
        # 外部可以根据 stage_indices 来提取每个 stage 的输出
        updated = {
            "inputs_embeds": inputs_embeds,
            "attention_mask_2d": mask_2d_bool.to(llm_device, dtype=torch.long),
            "attention_mask_4d": attn_bias,
            "position_ids": position_ids,
            "stage_indices": stage_indices,          # 传出 stage_indices
            "final_tgt_output": final_tgt_output,    # 按 stage 顺序的输出
        }
        
        if return_full_hidden:
            return final_tgt_output, updated, outputs.hidden_states
        else:
            return final_tgt_output, updated

    def forward_queries_parallel_v2(
        self,
        base_inputs_embeds: torch.Tensor,           # [B, Q_context, D]  视觉+导航上下文
        query_embeds: torch.Tensor,                 # [B, Q, D]  所有时间步的 queries（已按stage顺序排列）
        stage_indices: List[List[int]],             # 多尺度索引，如 [[5], [2,4], [1,3,5], [0,1,2,3,4,5]]
        attention_mask: Optional[torch.Tensor] = None,  # [B, Q_context] base 的 padding mask
        position_ids: Optional[torch.LongTensor] = None,
        training: bool = False,
        detach: bool = False,
        return_full_hidden: bool = False,
        use_time_position_encoding: bool = False,
        ablation_mask = 'none', # casual bidirectional
    ):
        """
        V4并行解码：简化版，不使用GT/预测点嵌入，仅context+stages_tgt
        
        核心改进：
        - 序列结构：[context, stage0_tgt, stage1_tgt, ..., stageN_tgt]
        - mask规则：
          1. context ↔ context: 双向可见
          2. context → stages: 禁止（context不看未来）
          3. stages → context: 可见（利用视觉信息）
          4. stage_i → stage_{i-1}: 可见（依赖上一层）
          5. stage_i ↔ stage_i: 双向可见
          6. stage_i → stage_j (j>i or j<i-1): 禁止（不看未来或更早层）
        
        Args:
            base_inputs_embeds: (B, Q_context, D) 视觉+导航上下文
            query_embeds: (B, Q, D) 所有时间步的query（按原始顺序，非stage顺序）
            stage_indices: 多尺度索引列表，如 [[5], [2,4], [1,3,5], [0,1,2,3,4,5]]
            attention_mask: (B, Q_context) context 的 padding mask
            training: 是否训练模式
        
        Returns:
            final_tgt_output: (B, Q_final, D) 按stage顺序的输出特征
            updated: dict，包含stage_indices、stage_len_list等信息
        """
        # ---- 形状检查 ----
        B, Q_context, D = base_inputs_embeds.shape
        Bq, T, Dq = query_embeds.shape
        assert B == Bq and D == Dq == self.model.config.hidden_size, \
            f"Shape mismatch: B={B}/{Bq}, D={D}/{Dq}/{self.model.config.hidden_size}"
        import os, ipdb
        if os.getenv("ACTION_DEBUG") == "1": ipdb.set_trace()         
        device = base_inputs_embeds.device
        llm = self.model
        llm_param = next(llm.parameters())
        llm_dtype = llm_param.dtype
        llm_device = llm_param.device
        NEG_INF = torch.finfo(llm_dtype).min
        POS_INF = torch.finfo(llm_dtype).max
        
        # ---- 数值清洗 ----
        base_inputs_embeds = sanitize_inputs_embeds(base_inputs_embeds)
        query_embeds = self.sanitize_tensor(query_embeds)
        
        # ---- 1. 重组queries按stage顺序 ----
        stage_tgt_list = []
        stage_len_list = []
        for i in range(len(stage_indices)):
            stage_tgt_list.append(query_embeds[:, stage_indices[i], :])
            stage_len_list.append(len(stage_indices[i]))
        final_tgt = torch.cat(stage_tgt_list, dim=1)  # (B, Q_final, D) torch.Size([2, 21, 4096])
        Q_final = final_tgt.shape[1]
        
        # 验证：Q_final应等于所有stage长度之和
        assert Q_final == sum(stage_len_list), \
            f"Q_final={Q_final} should equal sum(stage_len_list)={sum(stage_len_list)}"
        
        # ---- 2. 拼接embeds：[context, stages_tgt] ----
        inputs_embeds = torch.cat([base_inputs_embeds, final_tgt], dim=1)  # [B, Q_context+Q_final, D]
        L = Q_context + Q_final
        
        # # ---- 3. 构造多尺度自回归掩码 ----
        # attn_mask_2d = torch.zeros((L, L), device=llm_device, dtype=llm_dtype)
        
        # # 3.1 context ↔ context: 双向可见（已经是0）
        
        # # 3.2 context → stages: 禁止
        # attn_mask_2d[:Q_context, Q_context:] = NEG_INF
        
        # # 3.3 stages → context: 允许（已经是0）
        
        # # 3.4 stages ↔ stages: 分层因果掩码
        # # stage_i 可以看到 stage_{i-1} 和自己，不能看到其他
        # tgt_start = Q_context
        # tgt_tgt_mask = torch.full((Q_final, Q_final), fill_value=NEG_INF, device=llm_device, dtype=llm_dtype)
        
        # row_offset = 0
        # for i in range(len(stage_indices)):
        #     n_rows = stage_len_list[i]  # stage i 的query数量
            
        #     if i == 0:
        #         # Stage 0: 只能看到自己
        #         visible_start = 0
        #         visible_end = stage_len_list[0]
        #     else:
        #         # Stage i (i>0): 可以看到stage_{i-1}和自己
        #         visible_start = sum(stage_len_list[:i-1])  # stage_{i-1}的起始位置
        #         visible_end = sum(stage_len_list[:i+1])     # stage_i的结束位置
            
        #     # 设置可见区域为0（其余保持NEG_INF）
        #     tgt_tgt_mask[row_offset:row_offset+n_rows, visible_start:visible_end] = 0.0
            
        #     row_offset += n_rows
        
        # attn_mask_2d[tgt_start:, tgt_start:] = tgt_tgt_mask
        
        # ---- 3. 构造注意力掩码（根据ablation_mask参数选择掩码类型）----
        if ablation_mask == 'causal':
            # 消融实验：纯因果掩码（标准下三角掩码）
            # 每个位置只能看到自己及之前的位置
            attn_mask_2d = torch.triu(
                torch.ones((L, L), device=llm_device, dtype=llm_dtype), 
                diagonal=1
            ) * NEG_INF
            
        elif ablation_mask == 'bidirectional':
            # 消融实验：纯双向掩码（全部可见）
            # 所有位置之间都可以互相看见
            attn_mask_2d = torch.zeros((L, L), device=llm_device, dtype=llm_dtype)
            
        else:  # ablation_mask == 'none' or 'multiscale' (默认)
            # 默认：多尺度自回归掩码（混合掩码设计）
            attn_mask_2d = torch.zeros((L, L), device=llm_device, dtype=llm_dtype)
            
            # 3.1 context ↔ context: 双向可见（已经是0）
            
            # 3.2 context → stages: 禁止
            attn_mask_2d[:Q_context, Q_context:] = NEG_INF
            
            # 3.3 stages → context: 允许（已经是0）
            
            # 3.4 stages ↔ stages: 分层因果掩码
            # stage_i 可以看到 stage_{i-1} 和自己，不能看到其他
            tgt_start = Q_context
            tgt_tgt_mask = torch.full((Q_final, Q_final), fill_value=NEG_INF, device=llm_device, dtype=llm_dtype)
            
            row_offset = 0
            for i in range(len(stage_indices)):
                n_rows = stage_len_list[i]  # stage i 的query数量
                
                if i == 0:
                    # Stage 0: 只能看到自己
                    visible_start = 0
                    visible_end = stage_len_list[0]
                else:
                    # Stage i (i>0): 可以看到stage_{i-1}和自己
                    visible_start = sum(stage_len_list[:i-1])  # stage_{i-1}的起始位置
                    visible_end = sum(stage_len_list[:i+1])     # stage_i的结束位置
                
                # 设置可见区域为0（其余保持NEG_INF）
                tgt_tgt_mask[row_offset:row_offset+n_rows, visible_start:visible_end] = 0.0
                
                row_offset += n_rows
            
            attn_mask_2d[tgt_start:, tgt_start:] = tgt_tgt_mask


        # ---- 4. 扩展到4D并融合padding mask ----
        attn_bias = attn_mask_2d.unsqueeze(0).unsqueeze(1).expand(B, 1, L, L).contiguous()
        
        # 处理context的padding mask
        if attention_mask is None:
            attention_mask = torch.ones(B, Q_context, dtype=torch.bool, device=device)
        else:
            attention_mask = attention_mask.to(device=device, dtype=torch.bool)
        
        # stages默认全部有效
        stages_mask = torch.ones(B, Q_final, dtype=torch.bool, device=device)
        mask_2d_bool = torch.cat([attention_mask, stages_mask], dim=1)  # (B, L)
        
        # 屏蔽列（key不可见）和行（query不能看）
        attn_bias = attn_bias.masked_fill((~mask_2d_bool).view(B, 1, 1, L), NEG_INF)
        attn_bias = attn_bias.masked_fill((~mask_2d_bool).view(B, 1, L, 1), NEG_INF)
        
        # ---- 5. "全行被屏蔽"保险（避免softmax NaN）----
        row_all_masked = (attn_bias <= (NEG_INF * 0.5)).all(dim=-1, keepdim=True)  # (B,1,L,1)
        eye = torch.eye(L, device=llm_device, dtype=llm_dtype).unsqueeze(0).unsqueeze(1).expand(B, 1, L, L)
        fix_row = torch.full_like(attn_bias, NEG_INF)
        fix_row = torch.where(eye.bool(), torch.zeros_like(attn_bias), fix_row)
        attn_bias = torch.where(row_all_masked, fix_row, attn_bias)
        
        # ---- 6. 数值清洗 ----
        attn_bias = torch.nan_to_num(attn_bias, nan=0.0, posinf=POS_INF, neginf=NEG_INF)
        
        if use_time_position_encoding:
            # ---- 7. position_ids（关键修复：targets 用“真实时间步”的位置编码）----
            # 我们先构造一个长度为 Q_final 的时间步索引表：例如 [[5],[2,4],[1,3,5],...] -> [5,2,4,1,3,5,...]
            # 这个 time_index 用来给每个 target token 指定“它实际代表的时间步 t”
            # 最终 position_ids = [0..Q_context-1, base_offset + time_index[0], base_offset + time_index[1], ...]
            # 其中 base_offset = Q_context 或者（如果外部传了 base 的 position_ids）使用它最后一个位置作为参考。

            # 1) 展平成一维列表（Python list -> Tensor），长度 = Q_final
            #    注意：stage_indices 是一个 List[List[int]]
            flat_time_index = []
            for idxs in stage_indices:
                flat_time_index.extend(idxs)  # 追加这个 stage 的时间步索引
            # 变成 (Q_final,) 的 LongTensor，放在同一 device 上
            tgt_time_index = torch.tensor(flat_time_index, device=device, dtype=torch.long)  # [Q_final]

            # 2) 基座 context 的 position_ids
            if position_ids is None:
                # base/context 的位置：0..Q_context-1
                base_pos = torch.arange(0, Q_context, device=device, dtype=torch.long).unsqueeze(0).repeat(B, 1)  # [B, Q_context]
                # 以 base 的最后一个位置为偏移，给 targets 赋“真实时间步位置”
                # 注意 +1 是为了让第 0 个时间步与 base 之间有一个明显的间隔；不加也可以，但建议加 1。
                base_last = base_pos[:, -1:]  # [B,1]
                pos_tgt = base_last + 1 + tgt_time_index.view(1, -1).expand(B, -1)  # [B, Q_final]，每个样本一致

                position_ids = torch.cat([base_pos, pos_tgt], dim=1)  # [B, Q_context + Q_final]
            else:
                # 如果外部已经给了 base 的 position_ids：
                #   - 当它只覆盖了 base（形状 [B, Q_context]）时，我们在此基础上扩展 targets；
                #   - 当它已经包含了 targets（形状 [B, Q_context + Q_final]）时，我们会“重写”targets 部分，
                #     保证 targets 的位置=base_last+1+真实时间步。
                assert position_ids.dim() == 2 and position_ids.size(0) == B, "Unexpected position_ids shape"

                if position_ids.size(1) == Q_context:
                    base_pos = position_ids.to(device=device, dtype=torch.long)                # [B, Q_context]
                    base_last = base_pos[:, -1:]                                              # [B, 1]
                    pos_tgt = base_last + 1 + tgt_time_index.view(1, -1).expand(B, -1)        # [B, Q_final]
                    position_ids = torch.cat([base_pos, pos_tgt], dim=1)                      # [B, L]
                else:
                    # 已含 targets 的情形：我们保留 base 的部分，只“覆盖” targets 的位置为时间步语义
                    assert position_ids.size(1) == (Q_context + Q_final), \
                        f"position_ids length must be Q_context({Q_context})+Q_final({Q_final})"
                    base_pos = position_ids[:, :Q_context].to(device=device, dtype=torch.long)  # [B, Q_context]
                    base_last = base_pos[:, -1:]                                                # [B, 1]
                    pos_tgt = base_last + 1 + tgt_time_index.view(1, -1).expand(B, -1)          # [B, Q_final]
                    position_ids = torch.cat([base_pos, pos_tgt], dim=1)    
        else:
            # ---- 7. position_ids（连续递增）----
            if position_ids is None:
                position_ids = torch.arange(0, L, dtype=torch.long, device=device).unsqueeze(0).repeat(B, 1)
            else:
                if position_ids.size(1) == Q_context:
                    last_pos = position_ids[:, -1:]
                    incr_stages = torch.arange(1, Q_final + 1, device=device, dtype=position_ids.dtype).unsqueeze(0)
                    position_ids = torch.cat([position_ids, last_pos + incr_stages], dim=1)
        
        # ---- 8. dtype/device对齐 ----
        inputs_embeds = inputs_embeds.to(device=llm_device, dtype=llm_dtype).contiguous()
        position_ids = position_ids.to(device=llm_device, dtype=torch.long)
        
        # ---- 9. 过模型（根据mask类型选择实现）----
        # 检查是否可以使用Flash Attention（仅在简单mask场景）
        use_flash = (
            os.getenv("FLASHUSE", "0") == "1" 
            and FLASH_ATTN_AVAILABLE 
            and ablation_mask in ['causal', 'bidirectional']
            # and not return_full_hidden  # flash路径暂不支持返回中间hidden states
        )
        
        if use_flash:
            # 使用Flash Attention加速路径
            logger.info(f"Using Flash Attention with mask type: {ablation_mask}")
            is_causal = (ablation_mask == 'causal')
            
            outputs = self.model.forward_with_flash_attention(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                attention_mask=mask_2d_bool,  # 使用2D bool mask
                is_causal=is_causal,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=return_full_hidden,
                return_dict=True,
            )
        else:
            # 使用标准4D mask路径
            if use_flash and not FLASH_ATTN_AVAILABLE:
                logger.warning("FLASHUSE=1 but Flash Attention is not available. Falling back to standard attention.")
            
            outputs = self.model.forward_support_4d(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attn_bias,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                output_hidden_states=True if return_full_hidden else False,
                return_dict=True,
            )
        
        last_hidden = outputs.last_hidden_state  # [B, L, D]
        last_hidden = torch.nan_to_num(last_hidden, nan=0.0, posinf=POS_INF, neginf=NEG_INF)
        
        # 提取stages对应的输出：[B, Q_final, D]
        final_tgt_output = last_hidden[:, Q_context:, :]
        
        if detach:
            final_tgt_output = final_tgt_output.detach()
        
        # 返回结果和辅助信息
        updated = {
            "inputs_embeds": inputs_embeds,
            "attention_mask_2d": mask_2d_bool.to(llm_device, dtype=torch.long),
            "attention_mask_4d": attn_bias,
            "position_ids": position_ids,
            "stage_indices": stage_indices,
            "stage_len_list": stage_len_list,
            "final_tgt_output": final_tgt_output,
        }
        
        if return_full_hidden:
            return final_tgt_output, updated, outputs.hidden_states if outputs.hidden_states is not None else final_tgt_output
        else:
            return final_tgt_output, updated
    
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        points: Optional[torch.Tensor] = None,
        trajectories: Optional[torch.Tensor] = None,
        ego_features: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            if trajectories is not None or points is not None:
                (
                    inputs,
                    position_ids,
                    attention_mask,
                    _,
                    inputs_embeds,
                    _,
                    _
                ) = self.prepare_inputs_labels_for_multimodal_traj(
                    inputs,
                    position_ids,
                    attention_mask,
                    None,
                    None,
                    images,
                    points,
                    trajectories,
                    ego_features,
                    image_sizes=image_sizes
                )
            else:
                (
                    inputs,
                    position_ids,
                    attention_mask,
                    _,
                    inputs_embeds,
                    _
                ) = self.prepare_inputs_labels_for_multimodal(
                    inputs,
                    position_ids,
                    attention_mask,
                    None,
                    None,
                    images,
                    image_sizes=image_sizes
                )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)


        inputs_embeds = sanitize_inputs_embeds(inputs_embeds)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )


    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)

def sanitize_inputs_embeds(inputs_embeds: torch.Tensor, eps: float = 1e-5):
    """
    检查并修复 inputs_embeds 中的 NaN、Inf 和异常值。
    """
    orig_dtype = inputs_embeds.dtype
    
    if torch.isnan(inputs_embeds).any():
        print("Warning: NaNs detected in inputs_embeds. Replacing with zeros.")
        inputs_embeds = torch.nan_to_num(inputs_embeds, nan=0.0)

    if torch.isinf(inputs_embeds).any():
        print("Warning: Infs detected in inputs_embeds. Clipping to finite range.")
        inputs_embeds = torch.clamp(inputs_embeds, min=-1e4, max=1e4)

    max_abs = torch.max(torch.abs(inputs_embeds))
    if max_abs > 1e4:
        print(f"Warning: Very large values detected in inputs_embeds (max={max_abs.item():.2e}). Clipping.")
        inputs_embeds = torch.clamp(inputs_embeds, min=-1e4, max=1e4)

    return inputs_embeds.to(dtype=orig_dtype)
