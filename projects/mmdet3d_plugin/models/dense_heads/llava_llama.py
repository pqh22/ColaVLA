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

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaModel,
    LlamaForCausalLM,
)
from transformers.models.llama.modeling_llama import LLAMA_INPUTS_DOCSTRING
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    BaseModelOutputWithPast,
)

# from transformers.utils import add_start_docstrings_to_model_forward
from .llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
import logging
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

logger = logging.getLogger(__name__)

# Flash Attention
FLASH_ATTN_AVAILABLE = False
try:
    from flash_attn import flash_attn_func

    FLASH_ATTN_AVAILABLE = True
    logger.info("Flash Attention is available and can be used with FLASHUSE=1")
except ImportError:
    logger.warning(
        "Flash Attention is not available. Install with: pip install flash-attn"
    )
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
        Flash Attention mask

                Args:
        inputs_embeds: (B, L, D)
        position_ids: (B, L) positionID
        attention_mask: (B, L) 2D padding mask Trueposition
        is_causal: causalmask
        use_cache: False
        output_attentions: False
        output_hidden_states: hidden states
        return_dict: returndictionary

                Returns:
                    BaseModelOutputWithPast
        """
        if not FLASH_ATTN_AVAILABLE:
            raise RuntimeError(
                "Flash Attention is not available. Please install flash-attn."
            )

        if use_cache:
            raise NotImplementedError(
                "Flash Attention path does not support use_cache=True"
            )

        if output_attentions:
            raise NotImplementedError(
                "Flash Attention does not return attention weights"
            )

        batch_size, seq_length, _ = inputs_embeds.shape
        hidden_states = inputs_embeds

        # RoPEcos/sin attentionget
        # Flash AttentionprocessRoPE ensureposition_ids

        all_hidden_states = () if output_hidden_states else None

        # processpadding mask Flash Attention
        # attention_mask: (B, L) bool tensor, Trueposition
        # Flash Attentionkey_padding_mask
        key_padding_mask = None
        if attention_mask is not None:
            # bool maskFlash Attention
            # Flash Attention: Truemask False
            # mask: True Falsepadding
            # Translated note.
            key_padding_mask = ~attention_mask.bool()  # (B, L), Truepaddingposition

        # Translated note.
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # getattentionmlp
            self_attn = decoder_layer.self_attn

            # === Attention with Flash Attention ===
            residual = hidden_states
            hidden_states = decoder_layer.input_layernorm(hidden_states)

            # Q, K, V
            bsz, q_len, _ = hidden_states.size()
            query_states = self_attn.q_proj(hidden_states)
            key_states = self_attn.k_proj(hidden_states)
            value_states = self_attn.v_proj(hidden_states)

            # Reshape: (B, L, num_heads, head_dim)
            query_states = query_states.view(
                bsz, q_len, self_attn.num_heads, self_attn.head_dim
            )
            key_states = key_states.view(
                bsz, q_len, self_attn.num_key_value_heads, self_attn.head_dim
            )
            value_states = value_states.view(
                bsz, q_len, self_attn.num_key_value_heads, self_attn.head_dim
            )

            # RoPE
            kv_seq_len = key_states.shape[1]
            # (B, num_heads, L, head_dim)RoPE
            query_states_rope = query_states.transpose(1, 2)
            key_states_rope = key_states.transpose(1, 2)
            value_states_rope = value_states.transpose(1, 2)

            cos, sin = self_attn.rotary_emb(value_states_rope, seq_len=kv_seq_len)

            query_states_rope, key_states_rope = apply_rotary_pos_emb(
                query_states_rope, key_states_rope, cos, sin, position_ids
            )

            # (B, L, num_heads, head_dim)
            query_states = query_states_rope.transpose(1, 2)
            key_states = key_states_rope.transpose(1, 2)
            value_states = value_states_rope.transpose(1, 2)

            # processGQA num_key_value_heads < num_heads repeat
            if self_attn.num_key_value_groups > 1:
                # repeat_kv: (B, L, num_kv_heads, head_dim) -> (B, L, num_heads, head_dim)
                key_states = key_states.repeat_interleave(
                    self_attn.num_key_value_groups, dim=2
                )
                value_states = value_states.repeat_interleave(
                    self_attn.num_key_value_groups, dim=2
                )

            # Flash Attention
            # flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
            #                 window_size=(-1, -1), alibi_slopes=None, deterministic=False)
            # (batch_size, seqlen, nheads, headdim)

            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout_p=0.0,
                softmax_scale=None,  # default1/sqrt(head_dim)
                causal=is_causal,
                # key_padding_mask flash-attn 2.3.2
                # cu_seqlenspadding
            )

            # padding mask processpaddingposition
            if key_padding_mask is not None:
                # paddingposition
                attn_output = attn_output.masked_fill(
                    key_padding_mask.unsqueeze(-1).unsqueeze(-1),  # (B, L, 1, 1)
                    0.0,
                )

            # Reshape(B, L, hidden_size)
            attn_output = attn_output.reshape(
                bsz, q_len, self_attn.num_heads * self_attn.head_dim
            )

            # Output
            attn_output = self_attn.o_proj(attn_output)
            attn_output = torch.nan_to_num(
                attn_output, nan=0.0, posinf=1e5, neginf=-1e5
            )

            # Residual
            hidden_states = residual + attn_output

            # === MLPpart ===
            residual = hidden_states
            hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
            hidden_states = decoder_layer.mlp(hidden_states)
            hidden_states = residual + hidden_states
            hidden_states = torch.nan_to_num(
                hidden_states, nan=0.0, posinf=1e5, neginf=-1e5
            )

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, None, all_hidden_states, None]
                if v is not None
            )

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

        # ----------- -----------
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

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
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # ----------- `decoder_attention_mask` -----------
        if attention_mask is None:
            # default 1
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )

        if attention_mask.dim() == 4:
            # expand 4-D mask:
            # ― bool True visible -> 0 / -inf additive mask
            # ― float/half default additive mask
            if attention_mask.dtype == torch.bool:
                attn_dtype = inputs_embeds.dtype
                attention_mask = attention_mask.to(attn_dtype)
                # True → 0.0   False → -inf
                attention_mask = (1.0 - attention_mask) * torch.finfo(attn_dtype).min
            # device dtype
            attention_mask = attention_mask.to(
                device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )
            combined_attention_mask = attention_mask  # shape
        else:
            # 2-D bool → causal + padding mask
            combined_attention_mask = self._prepare_decoder_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )

        # **** `attention_mask` `combined_attention_mask`
        hidden_states = inputs_embeds
        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`"
            )
            use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

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
            hidden_states = torch.nan_to_num(
                hidden_states, nan=0.0, posinf=1e5, neginf=-1e5
            )

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
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
            number_tokens = list(range(32000, 32256))
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
            number_tokens += [28544, 11255, 1563, 1266, 4151, 523, 3396, 2408]
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
    # switchLoRA adapter LoRA

    #     Args:
    # adapter_name: adapter "lora_1" "lora_2"
    #     """
    #     if hasattr(self, 'set_adapter'):
    #         self.set_adapter(adapter_name)
    #         print(f"Switched to LoRA adapter: {adapter_name}")
    #     else:
    #         print("Warning: Model does not have multiple adapters")

    # def enable_lora_adapters(self, adapter_names: list):
    #     """
    # enableLoRA adapters LoRA

    #     Args:
    # adapter_names: adapter ["lora_1", "lora_2"]
    #     """
    #     if hasattr(self, 'set_adapter'):
    #         self.set_adapter(adapter_names)
    #         print(f"Enabled LoRA adapters: {adapter_names}")
    #     else:
    #         print("Warning: Model does not have multiple adapters")

    def get_active_adapters(self):
        """
        getLoRA adapters

                Returns:
        adapter
        """
        if hasattr(self, "active_adapters"):
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
                    inputs_embeds,  # labels
                    labels,  # labels imagepartIGNORE
                ) = self.prepare_inputs_labels_for_multimodal_traj(
                    input_ids,
                    position_ids,  # None
                    attention_mask,
                    past_key_values,
                    labels,
                    images,
                    points,
                    trajectories,
                    ego_features,  # None
                    image_sizes,
                )

            else:
                (
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels,
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    images,
                    image_sizes,
                )

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

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
            lm_head_slices = self.lm_head.weight.split(
                self.vocab_size // self.pretraining_tp, dim=0
            )
            logits = [
                F.linear(hidden_states, lm_head_slices[i])
                for i in range(self.pretraining_tp)
            ]
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

    # llava_llama.py LlavaLlamaForCausalLM

    def sanitize_tensor(self, x: torch.Tensor, eps: float = 1e-5):
        # sanitize
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
        base_inputs_embeds: torch.Tensor,  # [B, T, D] (/)
        query_embeds: torch.Tensor,  # [B, Q, D] ( queries)
        attention_mask: torch.Tensor = None,  # [B, T] [B, T+Q] +padding=False
        position_ids: torch.LongTensor = None,  # [B, T] [B, T+Q]
        past_key_values: list = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        detach: bool = False,
        return_full_hidden: bool = False,
        use_text_context: bool = False,
        text_embs_stage: str = "initial",  # initial middle last
    ):
        """
        training
        - attention_mask + padding False
        - queries (Q1 + A1 + Q2) padding
        - base queries base causal
        - padding note key query
        return:
        query_feats: [B, Q, D]
        updated: dict concatenate 2D/4D maskposition
        () all_hidden_states
        """

        # ---- shapecheck ----
        B, T, D = base_inputs_embeds.shape
        Bq, Q, Dq = query_embeds.shape
        assert B == Bq, "Batch size mismatch."
        assert D == Dq == self.model.config.hidden_size, (
            f"Hidden size mismatch: base={D}, query={Dq}, model={self.model.config.hidden_size}"
        )

        device = base_inputs_embeds.device
        llm = self.model
        llm_param = next(llm.parameters())
        llm_dtype = llm_param.dtype
        llm_device = llm_param.device
        NEG_INF = torch.finfo(llm_dtype).min
        POS_INF = torch.finfo(llm_dtype).max

        # ---- numericalsanitize ----
        base_inputs_embeds = sanitize_inputs_embeds(base_inputs_embeds)
        query_embeds = self.sanitize_tensor(query_embeds)

        # ---- concatenate ----
        inputs_embeds = torch.cat(
            [base_inputs_embeds, query_embeds], dim=1
        )  # [B, T+Q, D]
        L = T + Q

        # ---- attention_mask [B, T+Q] queries default(True) ----
        if attention_mask is None:
            attention_mask = torch.ones(B, T, dtype=torch.long, device=device)
        assert attention_mask.dim() == 2 and attention_mask.size(0) == B
        if attention_mask.size(1) == T:
            q_mask = torch.ones(B, Q, dtype=attention_mask.dtype, device=device)
            attention_mask = torch.cat([attention_mask, q_mask], dim=1)  # [B, T+Q]
        else:
            assert attention_mask.size(1) == L, "Unexpected attention_mask length."

        # ---- position_ids [B, T+Q] ----
        if position_ids is None:
            position_ids = (
                torch.arange(0, L, dtype=torch.long, device=device)
                .unsqueeze(0)
                .repeat(B, 1)
            )
        else:
            assert position_ids.dim() == 2 and position_ids.size(0) == B
            if position_ids.size(1) == T:
                last_pos = position_ids[:, -1:]  # [B,1]
                incr = torch.arange(
                    1, Q + 1, device=device, dtype=position_ids.dtype
                ).unsqueeze(0)  # [1,Q]
                position_ids = torch.cat(
                    [position_ids, last_pos + incr], dim=1
                )  # [B,T+Q]
            else:
                assert position_ids.size(1) == L, "Unexpected position_ids length."

        L = T + Q

        # ---- 1) build base causal forbidden base→query ----
        # base↔base bidirectional causal
        base_base = torch.zeros((T, T), device=llm_device, dtype=llm_dtype)

        # base→query forbidden
        base_to_query = torch.full(
            (T, Q), fill_value=NEG_INF, device=llm_device, dtype=llm_dtype
        )

        # query→base
        query_to_base = torch.zeros((Q, T), device=llm_device, dtype=llm_dtype)

        # query↔query bidirectional
        # query causal torch.triu(torch.ones((Q,Q), ...), diagonal=1)
        query_query = torch.zeros((Q, Q), device=llm_device, dtype=llm_dtype)

        upper = torch.cat([base_base, base_to_query], dim=1)  # [T, L]
        lower = torch.cat([query_to_base, query_query], dim=1)  # [Q, L]
        attn_bias = torch.cat([upper, lower], dim=0)  # [L, L]

        # expand 4D: [B, 1, L, L]
        attn_bias = attn_bias.unsqueeze(0).unsqueeze(1).expand(B, 1, L, L).contiguous()

        # ---- 2) merge 2D padding mask ----
        mask_2d_bool = attention_mask.to(device=llm_device, dtype=torch.bool)  # [B, L]

        # key
        attn_bias = attn_bias.masked_fill((~mask_2d_bool).view(B, 1, 1, L), NEG_INF)

        # position
        attn_bias = attn_bias.masked_fill((~mask_2d_bool).view(B, 1, L, 1), NEG_INF)

        # ---- 3) (=0 =NEG_INF) softmax NaN ----
        # NEG_INF softmax NaN padding
        row_all_masked = (attn_bias <= (NEG_INF * 0.5)).all(
            dim=-1, keepdim=True
        )  # [B,1,L,1]

        eye = torch.eye(L, device=llm_device, dtype=llm_dtype)  # [L,L]
        eye = eye.unsqueeze(0).unsqueeze(1).expand(B, 1, L, L)  # [B,1,L,L]

        fix_row = torch.full_like(attn_bias, NEG_INF)  # NEG_INF
        fix_row = torch.where(eye.bool(), torch.zeros_like(attn_bias), fix_row)  # 0

        attn_bias = torch.where(row_all_masked, fix_row, attn_bias)
        # ---- 4) NaN / ±Inf ----
        attn_bias = torch.nan_to_num(attn_bias, nan=0.0, posinf=POS_INF, neginf=NEG_INF)

        # ---- dtype / device ----
        inputs_embeds = inputs_embeds.to(
            device=llm_device, dtype=llm_dtype
        ).contiguous()
        position_ids = position_ids.to(device=llm_device, dtype=torch.long)

        # ---- 4D mask ----
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

        last_hidden = outputs.last_hidden_state  # [B, T+Q, D]
        last_hidden = torch.nan_to_num(
            last_hidden, nan=0.0, posinf=POS_INF, neginf=NEG_INF
        )

        query_feats = last_hidden[:, -Q:, :]  # [B, Q, D]

        if detach:
            query_feats = query_feats.detach()

        # Translated note.
        updated = {
            "inputs_embeds": inputs_embeds,
            "attention_mask_2d": attention_mask.to(
                llm_device, dtype=torch.long
            ),  # [B,L]
            "attention_mask_4d": attn_bias,  # [B,1,L,L]
            "position_ids": position_ids,
            "past_key_values": None,  # trainingenable
            "last_hidden": last_hidden,
            # "hidden_states": outputs.hidden_states,
        }
        import os, ipdb

        if os.getenv("HIDDEN_DEBUG") == "1":
            ipdb.set_trace()
        if return_full_hidden:
            return query_feats, updated, outputs.hidden_states
        else:
            return query_feats, updated

    def forward_queries_test(
        self,
        base_inputs_embeds: torch.Tensor,  # [B, T, D] eLLM image//
        query_embeds: torch.Tensor,  # [B, Q, D] queries
        attention_mask: Optional[torch.Tensor] = None,  # [B, T] [B, T+Q]
        position_ids: Optional[torch.LongTensor] = None,  # [B, T] [B, T+Q]
        past_key_values: Optional[list] = None,  # forward
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        detach: bool = False,  # True
        return_full_hidden: bool = False,  # hidden
        use_text_context: bool = False,
        text_embs_stage: str = "initial",  # initial middle last
    ):
        """
        learnable queries LLM
        return query position MLP
        logits loss

        return:
        query_feats: [B, Q, D] query token hidden
        updated: dict concat inputs_embeds/attention_mask/position_ids
        () all_hidden_states: return_full_hidden=True return
        """
        # import os, ipdb
        # if os.getenv("DEBUG") == "1": ipdb.set_trace()
        # check
        B, T, D = base_inputs_embeds.shape
        Bq, Q, Dq = query_embeds.shape
        assert B == Bq, (
            "Batch size mismatch between base_inputs_embeds and query_embeds"
        )
        assert D == Dq == self.model.config.hidden_size, (
            f"Hidden size mismatch: got D={D}, query D={Dq}, model={self.model.config.hidden_size}"
        )

        device = base_inputs_embeds.device

        # processnumerical
        base_inputs_embeds = sanitize_inputs_embeds(base_inputs_embeds)  # NaN
        query_embeds = self.sanitize_tensor(query_embeds)  # NaN

        # concatenate embeds
        inputs_embeds = torch.cat(
            [base_inputs_embeds, query_embeds], dim=1
        )  # [B, T+Q, D]
        # inputs_embeds = self.query_norm(inputs_embeds.float()).to(dtype=base_inputs_embeds.dtype)

        # attention mask
        if attention_mask is None:  # None
            attention_mask = torch.ones(B, T, dtype=torch.long, device=device)
        if attention_mask.size(1) != T:
            # Q
            assert attention_mask.size(1) in (T, T + Q), (
                "Unexpected attention_mask length"
            )
        if attention_mask.size(1) == T:
            q_mask = torch.ones(B, Q, dtype=attention_mask.dtype, device=device)
            attention_mask = torch.cat([attention_mask, q_mask], dim=1)  # [B, T+Q]

        # position ids
        if position_ids is None:
            # position 0..T+Q-1
            position_ids = (
                torch.arange(0, T + Q, dtype=torch.long, device=device)
                .unsqueeze(0)
                .repeat(B, 1)
            )
        else:
            assert position_ids.size(1) in (T, T + Q), "Unexpected position_ids length"
            if position_ids.size(1) == T:
                # position Q position
                last_pos = position_ids[:, -1:]  # [B,1]
                incr = torch.arange(
                    1, Q + 1, device=device, dtype=position_ids.dtype
                ).unsqueeze(0)  # [1,Q]
                pos_q = last_pos + incr
                position_ids = torch.cat([position_ids, pos_q], dim=1)  # [B, T+Q]

        llm = self.model
        llm_param = next(llm.parameters())
        llm_dtype = llm_param.dtype
        llm_device = llm_param.device
        NEG_INF = torch.finfo(llm_dtype).min
        POS_INF = torch.finfo(llm_dtype).max

        L = T + Q

        # ---- 1) build base causal forbidden base→query ----
        # base↔base bidirectional causal
        base_base = torch.zeros((T, T), device=llm_device, dtype=llm_dtype)

        # base→query forbidden
        base_to_query = torch.full(
            (T, Q), fill_value=NEG_INF, device=llm_device, dtype=llm_dtype
        )

        # query→base
        query_to_base = torch.zeros((Q, T), device=llm_device, dtype=llm_dtype)

        # query↔query bidirectional
        # query causal torch.triu(torch.ones((Q,Q), ...), diagonal=1)
        query_query = torch.zeros((Q, Q), device=llm_device, dtype=llm_dtype)

        upper = torch.cat([base_base, base_to_query], dim=1)  # [T, L]
        lower = torch.cat([query_to_base, query_query], dim=1)  # [Q, L]
        attn_bias = torch.cat([upper, lower], dim=0)  # [L, L]

        # expand 4D: [B, 1, L, L]
        attn_bias = attn_bias.unsqueeze(0).unsqueeze(1).expand(B, 1, L, L).contiguous()

        # ---- 2) merge 2D padding mask ----
        mask_2d_bool = attention_mask.to(device=llm_device, dtype=torch.bool)  # [B, L]

        # key
        attn_bias = attn_bias.masked_fill((~mask_2d_bool).view(B, 1, 1, L), NEG_INF)

        # position
        attn_bias = attn_bias.masked_fill((~mask_2d_bool).view(B, 1, L, 1), NEG_INF)

        # ---- 3) (=0 =NEG_INF) softmax NaN ----
        # NEG_INF softmax NaN padding
        row_all_masked = (attn_bias <= (NEG_INF * 0.5)).all(
            dim=-1, keepdim=True
        )  # [B,1,L,1]

        eye = torch.eye(L, device=llm_device, dtype=llm_dtype)  # [L,L]
        eye = eye.unsqueeze(0).unsqueeze(1).expand(B, 1, L, L)  # [B,1,L,L]

        fix_row = torch.full_like(attn_bias, NEG_INF)  # NEG_INF
        fix_row = torch.where(eye.bool(), torch.zeros_like(attn_bias), fix_row)  # 0

        attn_bias = torch.where(row_all_masked, fix_row, attn_bias)

        # ---- 4) NaN / ±Inf ----
        attn_bias = torch.nan_to_num(attn_bias, nan=0.0, posinf=POS_INF, neginf=NEG_INF)

        # note sanitize dtype float32 LLM dtype
        inputs_embeds = inputs_embeds.to(
            device=llm_device, dtype=llm_dtype
        ).contiguous()

        # query_embeds process
        # query_embeds process ensure llm_dtype
        # query_embeds = query_embeds.to(device=llm_device, dtype=llm_dtype)

        # padding key 4D mask
        if attention_mask is None or attention_mask.size(1) != L:
            # [B,T] [B,L]
            pass
        key_pad = (1 - attention_mask).to(dtype=llm_dtype, device=llm_device)  # 1 -inf
        attn_bias = attn_bias + key_pad.view(B, 1, 1, L) * NEG_INF

        # mask & pos dtype mask long/bool pos long
        if attention_mask is None:
            attention_mask = torch.ones(
                inputs_embeds.size()[:2], device=llm_device, dtype=torch.long
            )
        else:
            attention_mask = attention_mask.to(device=llm_device, dtype=torch.long)

        if position_ids is None:
            B, TQ, _ = inputs_embeds.shape
            position_ids = (
                torch.arange(0, TQ, device=llm_device, dtype=torch.long)
                .unsqueeze(0)
                .expand(B, -1)
            )
        else:
            position_ids = position_ids.to(device=llm_device, dtype=torch.long)

        # --- 4D mask default ---
        outputs = self.model.forward_support_4d(
            input_ids=None,
            inputs_embeds=inputs_embeds.to(device=llm_device, dtype=llm_dtype),
            attention_mask=attn_bias,  # 4D mask
            position_ids=position_ids.to(llm_device),
            past_key_values=None,  # pkv
            use_cache=False,  # cache bidirectional
            output_hidden_states=True,
            return_dict=True,
        )

        last_hidden = outputs.last_hidden_state  # [B, T+Q, D]
        last_hidden = torch.nan_to_num(
            last_hidden, nan=0.0, posinf=POS_INF, neginf=NEG_INF
        )
        query_feats = last_hidden[:, -Q:, :]  # [B, Q, D]

        if detach:
            query_feats = query_feats.detach()

        updated = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": outputs.past_key_values,
            "last_hidden": last_hidden,
        }
        import os, ipdb

        if os.getenv("HIDDEN_DEBUG") == "1":
            ipdb.set_trace()
        if return_full_hidden:
            return query_feats, updated, outputs.hidden_states
        else:
            return query_feats, updated

    def forward_queries(
        self,
        base_inputs_embeds: torch.Tensor,  # [B, T, D]
        query_embeds: torch.Tensor,  # [B, Q, D] queries
        attention_mask: Optional[torch.Tensor] = None,  # [B, T] [B, T+Q]
        position_ids: Optional[torch.LongTensor] = None,  # [B, T] [B, T+Q]
        past_key_values: Optional[list] = None,  # forward
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        detach: bool = False,  # True
        return_full_hidden: bool = False,  # hidden
        training: bool = False,
        use_text_context: bool = False,
        text_embs_stage: str = "initial",  # initial middle last
    ):
        if training:
            return self.forward_queries_train(
                base_inputs_embeds,
                query_embeds,
                attention_mask,
                position_ids,
                past_key_values,
                use_cache,
                output_attentions,
                output_hidden_states,
                detach,
                return_full_hidden,
                use_text_context,
                text_embs_stage,
            )
        else:
            return self.forward_queries_test(
                base_inputs_embeds,
                query_embeds,
                attention_mask,
                position_ids,
                past_key_values,
                use_cache,
                output_attentions,
                output_hidden_states,
                detach,
                return_full_hidden,
                use_text_context,
                text_embs_stage,
            )

    def forward_queries_stage_inference(
        self,
        base_context: torch.Tensor,  # (B, Q_context, D) +
        traj_embeds_slots: torch.Tensor,  # (B, T, D) T visibleposition
        prev_stage_query_outs: Optional[
            torch.Tensor
        ],  # (B, K_all_prev, D) stagesquery outputs
        curr_stage_query: torch.Tensor,  # (B, n_curr, D) stagequeries
        last_stage_len: int = 0,  # stage mask ensureStage i-1
        traj_valid_mask: Optional[torch.Tensor] = None,  # (B, T) bool mask trajposition
        attention_mask: Optional[
            torch.Tensor
        ] = None,  # (B, Q_context) basepadding mask
        position_ids: Optional[torch.LongTensor] = None,
    ):
        """
        inferencestageforward trainingmaskposition

        training [base, gt_traj_embeds, all_prev_stages_queries, curr_stage_queries]
        [base_context, traj_embeds_slots, prev_stage_query_outs(), curr_stage_query]
                    [Q_context,    T,                 K_all_prev,                      n_curr]

        Translated note.
        - ** stages** query outputs Stage i-1 ensurepositiontraining
        - last_stage_len mask ensure curr Stage i-1 stage
        - position

        mask training
        1. base ↔ base: bidirectionalvisible
        2. traj_slots ↔ traj_slots: visible
        3. base → (traj + prev + curr): forbidden
        4. traj → base: visible
        5. traj → (prev + curr): forbidden
        6. prev → base: visible
        7. prev → traj: visible
        8. prev ↔ prev: bidirectionalvisible curr last_stage_len Stage i-1
        9. prev → curr: forbidden
        10. curr → base: visible
        11. curr → traj: visible stage
        12. curr → prev: last_stage_len Stage i-1 training
        13. curr ↔ curr: bidirectionalvisible

                Args:
        base_context: (B, Q_context, D)
        traj_embeds_slots: T visibleposition (B, T, D)
        prev_stage_query_outs: stagesquery outputs (B, K_all_prev, D) None
        curr_stage_query: stagequeries (B, n_curr, D)
        last_stage_len: stage Stage i-1 maskcurrStage i-1
        traj_valid_mask: traj_embeds_slotsposition (B, T) True
        attention_mask: basepadding mask (B, Q_context)

                Returns:
                    curr_query_out: (B, n_curr, D)
        """
        B, Q_context, D = base_context.shape
        _, T, _ = traj_embeds_slots.shape
        _, n_curr, _ = curr_stage_query.shape
        K_prev = (
            prev_stage_query_outs.shape[1] if prev_stage_query_outs is not None else 0
        )

        # traj_valid_mask defaultposition
        if traj_valid_mask is None:
            traj_valid_mask = torch.ones(
                B, T, dtype=torch.bool, device=base_context.device
            )

        device = base_context.device
        llm = self.model
        llm_param = next(llm.parameters())
        llm_dtype = llm_param.dtype
        llm_device = llm_param.device
        NEG_INF = torch.finfo(llm_dtype).min
        POS_INF = torch.finfo(llm_dtype).max

        # ---- numericalsanitize ----
        base_context = sanitize_inputs_embeds(base_context)
        traj_embeds_slots = self.sanitize_tensor(traj_embeds_slots)
        curr_stage_query = self.sanitize_tensor(curr_stage_query)
        if prev_stage_query_outs is not None:
            prev_stage_query_outs = self.sanitize_tensor(prev_stage_query_outs)

        # ---- concatenate embeds ----
        if prev_stage_query_outs is not None:
            inputs_embeds = torch.cat(
                [
                    base_context,
                    traj_embeds_slots,
                    prev_stage_query_outs,
                    curr_stage_query,
                ],
                dim=1,
            )
        else:
            inputs_embeds = torch.cat(
                [base_context, traj_embeds_slots, curr_stage_query], dim=1
            )
        L = Q_context + T + K_prev + n_curr

        # ---- buildmask training ----
        attn_mask_2d = torch.zeros((L, L), device=llm_device, dtype=llm_dtype)

        # 1. traj_slots ↔ traj_slots: visible
        traj_start = Q_context
        traj_end = Q_context + T
        traj_block = torch.full(
            (T, T), fill_value=NEG_INF, device=llm_device, dtype=llm_dtype
        )
        diag_idx = torch.arange(T, device=llm_device)
        traj_block[diag_idx, diag_idx] = 0.0
        attn_mask_2d[traj_start:traj_end, traj_start:traj_end] = traj_block

        # 2. base → traj + prev + curr : forbidden
        attn_mask_2d[:Q_context, Q_context:] = NEG_INF

        # 3. traj → prev: forbidden
        if K_prev > 0:
            prev_start = Q_context + T
            prev_end = Q_context + T + K_prev
            attn_mask_2d[traj_start:traj_end, prev_start:prev_end] = NEG_INF

        # 4. traj → curr: forbidden
        curr_start = Q_context + T + K_prev
        attn_mask_2d[traj_start:traj_end, curr_start:] = NEG_INF

        # 5. prev → curr: forbidden prev
        if K_prev > 0:
            attn_mask_2d[prev_start:prev_end, curr_start:] = NEG_INF

        # 6. curr → prev: last_stage_len Stage i-1 training
        if K_prev > 0 and last_stage_len > 0 and last_stage_len < K_prev:
            # forbidden curr prev (K_prev - last_stage_len) position
            attn_mask_2d[
                curr_start:, prev_start : prev_start + (K_prev - last_stage_len)
            ] = NEG_INF

        # defaultvisible 0

        # ---- expand 4D merge padding mask ----
        attn_bias = (
            attn_mask_2d.unsqueeze(0).unsqueeze(1).expand(B, 1, L, L).contiguous()
        )

        # process base padding mask
        if attention_mask is None:
            attention_mask = torch.ones(B, Q_context, dtype=torch.bool, device=device)
        else:
            attention_mask = attention_mask.to(device=device, dtype=torch.bool)

        # traj: traj_valid_mask position
        # prev, curr: default
        traj_mask = traj_valid_mask.to(device=device, dtype=torch.bool)  # mask
        prev_mask = (
            torch.ones(B, K_prev, dtype=torch.bool, device=device)
            if K_prev > 0
            else None
        )
        curr_mask = torch.ones(B, n_curr, dtype=torch.bool, device=device)

        if prev_mask is not None:
            mask_2d_bool = torch.cat(
                [attention_mask, traj_mask, prev_mask, curr_mask], dim=1
            )
        else:
            mask_2d_bool = torch.cat([attention_mask, traj_mask, curr_mask], dim=1)

        # keyvisible query
        attn_bias = attn_bias.masked_fill((~mask_2d_bool).view(B, 1, 1, L), NEG_INF)
        attn_bias = attn_bias.masked_fill((~mask_2d_bool).view(B, 1, L, 1), NEG_INF)

        # "" softmax NaN
        row_all_masked = (attn_bias <= (NEG_INF * 0.5)).all(dim=-1, keepdim=True)
        eye = (
            torch.eye(L, device=llm_device, dtype=llm_dtype)
            .unsqueeze(0)
            .unsqueeze(1)
            .expand(B, 1, L, L)
        )
        fix_row = torch.full_like(attn_bias, NEG_INF)
        fix_row = torch.where(eye.bool(), torch.zeros_like(attn_bias), fix_row)
        attn_bias = torch.where(row_all_masked, fix_row, attn_bias)

        # numericalsanitize
        attn_bias = torch.nan_to_num(attn_bias, nan=0.0, posinf=POS_INF, neginf=NEG_INF)

        # ---- position_ids trainingposition ----
        # trainingposition [0...Q_context-1, Q_context...Q_context+T-1, Q_context+T...Q_context+T+K_prev-1, Q_context+T+K_prev...]
        # trainingposition
        if position_ids is None:
            # trainingposition
            # base: 0...Q_context-1
            # traj: Q_context...Q_context+T-1
            # prev: Q_context+T...Q_context+T+K_prev-1
            # curr: Q_context+T+K_prev...Q_context+T+K_prev+n_curr-1
            position_ids = (
                torch.arange(0, L, dtype=torch.long, device=device)
                .unsqueeze(0)
                .repeat(B, 1)
            )
        else:
            # base position_ids expand
            if position_ids.size(1) == Q_context:
                # trainingexpand
                last_pos = position_ids[:, -1:]
                # traj part last_pos + 1 last_pos + T
                incr_traj = torch.arange(
                    1, T + 1, device=device, dtype=position_ids.dtype
                ).unsqueeze(0)
                pos_traj = last_pos + incr_traj

                # prev part last_pos + T + 1 last_pos + T + K_prev
                incr_prev = torch.arange(
                    T + 1, T + K_prev + 1, device=device, dtype=position_ids.dtype
                ).unsqueeze(0)
                pos_prev = last_pos + incr_prev if K_prev > 0 else None

                # curr part last_pos + T + K_prev + 1 last_pos + T + K_prev + n_curr
                incr_curr = torch.arange(
                    T + K_prev + 1,
                    T + K_prev + n_curr + 1,
                    device=device,
                    dtype=position_ids.dtype,
                ).unsqueeze(0)
                pos_curr = last_pos + incr_curr

                # concatenate
                if pos_prev is not None:
                    position_ids = torch.cat(
                        [position_ids, pos_traj, pos_prev, pos_curr], dim=1
                    )
                else:
                    position_ids = torch.cat([position_ids, pos_traj, pos_curr], dim=1)

        # ---- dtype/device ----
        inputs_embeds = inputs_embeds.to(
            device=llm_device, dtype=llm_dtype
        ).contiguous()
        position_ids = position_ids.to(device=llm_device, dtype=torch.long)

        # ---- 4Dmask ----
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
        last_hidden = torch.nan_to_num(
            last_hidden, nan=0.0, posinf=POS_INF, neginf=NEG_INF
        )

        # curr_stage_query
        curr_query_out = last_hidden[:, -n_curr:, :]  # (B, n_curr, D)

        return curr_query_out

    def forward_queries_parallel(
        self,
        base_inputs_embeds: torch.Tensor,  # [B, T, D]
        query_embeds: torch.Tensor,  # [B, Q, D] queries
        stage_indices: List[List[int]],  # [[5], [2,4], [1,3,5], [0,1,2,3,4,5]]
        gt_traj_embeds: Optional[torch.Tensor] = None,  # (B, T, d_in)
        attention_mask: Optional[torch.Tensor] = None,  # [B, T] base padding mask
        position_ids: Optional[torch.LongTensor] = None,
        training: bool = False,
        detach: bool = False,
        return_full_hidden: bool = False,
    ):
        """
        causalmaskensure

        core VAR
        - stage level
        - buildmask i <= + base
        - for

                Args:
        base_inputs_embeds: (B, T, D) +
        query_embeds: (B, Q, D) query position+
        stage_indices: [[5], [2,4], [1,3,5], [0,1,2,3,4,5]]
        gt_traj_embeds: (B, T, d_in)
        attention_mask: (B, T) base padding mask
        training: training

                Returns:
        query_feats: (B, Q, D)
        updated: dict concatenate inputs_embeds/attention_mask
        """
        # ---- shapecheck ----
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

        # ---- numericalsanitize ----
        base_inputs_embeds = sanitize_inputs_embeds(base_inputs_embeds)
        query_embeds = self.sanitize_tensor(query_embeds)
        import os, ipdb

        if os.getenv("DEBUG") == "1":
            ipdb.set_trace()
        # tgt reconstruction
        stage_tgt_list = []
        stage_len_list = []
        for i in range(len(stage_indices)):
            stage_tgt_list.append(query_embeds[:, stage_indices[i], :])
            stage_len_list.append(len(stage_indices[i]))
        final_tgt = torch.cat(stage_tgt_list, dim=1)
        assert final_tgt.shape[1] == sum(stage_len_list)
        Q_final = final_tgt.shape[1]

        # index list: stage inputs_embeds position
        final_index_list = []
        for i in range(len(stage_len_list)):
            final_index_list.append(Q_context + T + sum(stage_len_list[:i]))

        # validate
        assert len(final_index_list) == len(stage_indices), (
            f"final_index_list length {len(final_index_list)} must equal stage_indices length {len(stage_indices)}"
        )

        # validate stage position
        for i in range(len(stage_indices)):
            expected_start = Q_context + T + sum(stage_len_list[:i])
            assert final_index_list[i] == expected_start, (
                f"Stage {i} start position {final_index_list[i]} must be {expected_start}"
            )

        # ---- 2. concatenate embeds ----
        inputs_embeds = torch.cat(
            [base_inputs_embeds, gt_traj_embeds, final_tgt], dim=1
        )  # [B, Q_context+T+Q_final, D]
        L = Q_context + T + Q_final

        # ---- 3. buildmask ----
        # mask [L, L] defaultvisible 0
        attn_mask_2d = torch.zeros((L, L), device=llm_device, dtype=llm_dtype)

        # 3.1 base GT part bidirectionalvisible 0
        # base ↔ base: 0
        # GT ↔ GT:
        gt_block = torch.full(
            (T, T), fill_value=NEG_INF, device=llm_device, dtype=llm_dtype
        )
        diag_idx = torch.arange(T, device=llm_device)
        gt_block[diag_idx, diag_idx] = 0.0
        attn_mask_2d[Q_context : Q_context + T, Q_context : Q_context + T] = gt_block

        # base ↔ GT:
        # 3.2 base → tgt: forbidden base query
        attn_mask_2d[:Q_context, Q_context:] = NEG_INF

        # 3.3 GT → tgt: forbidden GT query
        attn_mask_2d[Q_context : Q_context + T, Q_context + T :] = NEG_INF

        # 3.4 tgt → base: 0

        # 3.5 tgt → GT: stage
        # defaultforbidden stage GT
        tgt_start = Q_context + T
        tgt_to_gt_mask = torch.full(
            (Q_final, T), fill_value=NEG_INF, device=llm_device, dtype=llm_dtype
        )

        # stage stage GT
        tgt_offset = 0
        for stage_id in range(len(stage_indices)):
            n_queries = stage_len_list[stage_id]  # stage query

            # Stage i stage_indices[i-1] GT
            if stage_id > 0:
                prev_stage_gt_indices = stage_indices[stage_id - 1]
                # GT positionvisible
                for gt_idx in prev_stage_gt_indices:
                    tgt_to_gt_mask[tgt_offset : tgt_offset + n_queries, gt_idx] = (
                        0.0  # torch.Size([21, 6])
                    )
            # else: Stage 0 GT NEG_INF

            tgt_offset += n_queries

        # tgt → GT maskmask
        attn_mask_2d[tgt_start:, Q_context : Q_context + T] = tgt_to_gt_mask

        # 3.6 tgt ↔ tgt: causalmask Stage i Stage i-1 Stage i
        # Translated note.
        # 1. Stage i Stage i-1
        # 2. Stage 5 attend Stage 0
        # 3. stage outputs
        tgt_tgt_mask = torch.full(
            (Q_final, Q_final), fill_value=NEG_INF, device=llm_device, dtype=llm_dtype
        )

        row_offset = 0
        for i in range(len(stage_indices)):
            n_rows = stage_len_list[i]  # Stage i query

            if i == 0:
                # Stage 0:
                visible_start = 0
                visible_end = stage_len_list[0]
            else:
                # Stage i (i>0): Stage i-1
                visible_start = sum(stage_len_list[: i - 1])  # Stage i-1 position
                visible_end = sum(stage_len_list[: i + 1])  # Stage i position

            # visible 0 NEG_INF
            tgt_tgt_mask[
                row_offset : row_offset + n_rows, visible_start:visible_end
            ] = 0.0

            row_offset += n_rows

        attn_mask_2d[tgt_start:, tgt_start:] = tgt_tgt_mask

        # ---- 4. expand 4D merge padding mask ----
        attn_bias = (
            attn_mask_2d.unsqueeze(0).unsqueeze(1).expand(B, 1, L, L).contiguous()
        )

        # process base padding mask
        if attention_mask is None:
            attention_mask = torch.ones(B, Q_context, dtype=torch.bool, device=device)
        else:
            attention_mask = attention_mask.to(device=device, dtype=torch.bool)

        # GT query partdefault
        gt_mask = torch.ones(B, T, dtype=torch.bool, device=device)
        q_mask = torch.ones(B, Q_final, dtype=torch.bool, device=device)
        mask_2d_bool = torch.cat([attention_mask, gt_mask, q_mask], dim=1)  # (B, L)

        # key visible query
        attn_bias = attn_bias.masked_fill((~mask_2d_bool).view(B, 1, 1, L), NEG_INF)
        attn_bias = attn_bias.masked_fill((~mask_2d_bool).view(B, 1, L, 1), NEG_INF)

        # ---- 5. "" softmax NaN ----
        row_all_masked = (attn_bias <= (NEG_INF * 0.5)).all(dim=-1, keepdim=True)
        eye = (
            torch.eye(L, device=llm_device, dtype=llm_dtype)
            .unsqueeze(0)
            .unsqueeze(1)
            .expand(B, 1, L, L)
        )
        fix_row = torch.full_like(attn_bias, NEG_INF)
        fix_row = torch.where(eye.bool(), torch.zeros_like(attn_bias), fix_row)
        attn_bias = torch.where(row_all_masked, fix_row, attn_bias)

        # ---- 6. numericalsanitize ----
        attn_bias = torch.nan_to_num(attn_bias, nan=0.0, posinf=POS_INF, neginf=NEG_INF)

        # ---- 7. position_ids ----
        if position_ids is None:
            position_ids = (
                torch.arange(0, L, dtype=torch.long, device=device)
                .unsqueeze(0)
                .repeat(B, 1)
            )
        else:
            if position_ids.size(1) == Q_context:
                last_pos = position_ids[:, -1:]
                incr_gt = torch.arange(
                    1, T + 1, device=device, dtype=position_ids.dtype
                ).unsqueeze(0)
                incr_q = torch.arange(
                    T + 1, T + Q_final + 1, device=device, dtype=position_ids.dtype
                ).unsqueeze(0)
                position_ids = torch.cat(
                    [position_ids, last_pos + incr_gt, last_pos + incr_q], dim=1
                )

        # ---- 8. dtype/device ----
        inputs_embeds = inputs_embeds.to(
            device=llm_device, dtype=llm_dtype
        ).contiguous()
        position_ids = position_ids.to(device=llm_device, dtype=torch.long)

        # ---- 9. 4D mask ----
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
        last_hidden = torch.nan_to_num(
            last_hidden, nan=0.0, posinf=POS_INF, neginf=NEG_INF
        )

        # final_tgt [B, Q_final, D]
        final_tgt_output = last_hidden[:, tgt_start:, :]  # [B, Q_final, D]

        if detach:
            final_tgt_output = final_tgt_output.detach()

        # return final_tgt_output stage
        # stage_indices stage
        updated = {
            "inputs_embeds": inputs_embeds,
            "attention_mask_2d": mask_2d_bool.to(llm_device, dtype=torch.long),
            "attention_mask_4d": attn_bias,
            "position_ids": position_ids,
            "stage_indices": stage_indices,  # stage_indices
            "final_tgt_output": final_tgt_output,  # stage
        }

        if return_full_hidden:
            return final_tgt_output, updated, outputs.hidden_states
        else:
            return final_tgt_output, updated

    def forward_queries_parallel_v2(
        self,
        base_inputs_embeds: torch.Tensor,  # [B, Q_context, D] +
        query_embeds: torch.Tensor,  # [B, Q, D] queries stage
        stage_indices: List[List[int]],  # [[5], [2,4], [1,3,5], [0,1,2,3,4,5]]
        attention_mask: Optional[
            torch.Tensor
        ] = None,  # [B, Q_context] base padding mask
        position_ids: Optional[torch.LongTensor] = None,
        training: bool = False,
        detach: bool = False,
        return_full_hidden: bool = False,
        use_time_position_encoding: bool = False,
        ablation_mask="none",  # casual bidirectional
    ):
        """
        V4 GT/ context+stages_tgt

        core
        - [context, stage0_tgt, stage1_tgt, ..., stageN_tgt]
        - mask
        1. context ↔ context: bidirectionalvisible
        2. context → stages: forbidden context
        3. stages → context: visible
        4. stage_i → stage_{i-1}: visible
        5. stage_i ↔ stage_i: bidirectionalvisible
        6. stage_i → stage_j (j>i or j<i-1): forbidden

                Args:
        base_inputs_embeds: (B, Q_context, D) +
        query_embeds: (B, Q, D) query original stage
        stage_indices: [[5], [2,4], [1,3,5], [0,1,2,3,4,5]]
        attention_mask: (B, Q_context) context padding mask
        training: training

                Returns:
        final_tgt_output: (B, Q_final, D) stage
        updated: dict stage_indices stage_len_list
        """
        # ---- shapecheck ----
        B, Q_context, D = base_inputs_embeds.shape
        Bq, T, Dq = query_embeds.shape
        assert B == Bq and D == Dq == self.model.config.hidden_size, (
            f"Shape mismatch: B={B}/{Bq}, D={D}/{Dq}/{self.model.config.hidden_size}"
        )
        import os, ipdb

        if os.getenv("ACTION_DEBUG") == "1":
            ipdb.set_trace()
        device = base_inputs_embeds.device
        llm = self.model
        llm_param = next(llm.parameters())
        llm_dtype = llm_param.dtype
        llm_device = llm_param.device
        NEG_INF = torch.finfo(llm_dtype).min
        POS_INF = torch.finfo(llm_dtype).max

        # ---- numericalsanitize ----
        base_inputs_embeds = sanitize_inputs_embeds(base_inputs_embeds)
        query_embeds = self.sanitize_tensor(query_embeds)

        # ---- 1. queriesstage ----
        stage_tgt_list = []
        stage_len_list = []
        for i in range(len(stage_indices)):
            stage_tgt_list.append(query_embeds[:, stage_indices[i], :])
            stage_len_list.append(len(stage_indices[i]))
        final_tgt = torch.cat(
            stage_tgt_list, dim=1
        )  # (B, Q_final, D) torch.Size([2, 21, 4096])
        Q_final = final_tgt.shape[1]

        # validate Q_finalstage
        assert Q_final == sum(stage_len_list), (
            f"Q_final={Q_final} should equal sum(stage_len_list)={sum(stage_len_list)}"
        )

        # ---- 2. concatenateembeds [context, stages_tgt] ----
        inputs_embeds = torch.cat(
            [base_inputs_embeds, final_tgt], dim=1
        )  # [B, Q_context+Q_final, D]
        L = Q_context + Q_final

        # # ---- 3. buildmask ----
        # attn_mask_2d = torch.zeros((L, L), device=llm_device, dtype=llm_dtype)

        # # 3.1 context ↔ context: bidirectionalvisible 0

        # # 3.2 context → stages: forbidden
        # attn_mask_2d[:Q_context, Q_context:] = NEG_INF

        # # 3.3 stages → context: 0

        # # 3.4 stages ↔ stages: causalmask
        # # stage_i stage_{i-1}
        # tgt_start = Q_context
        # tgt_tgt_mask = torch.full((Q_final, Q_final), fill_value=NEG_INF, device=llm_device, dtype=llm_dtype)

        # row_offset = 0
        # for i in range(len(stage_indices)):
        # n_rows = stage_len_list[i] # stage i query

        #     if i == 0:
        # # Stage 0:
        #         visible_start = 0
        #         visible_end = stage_len_list[0]
        #     else:
        # # Stage i (i>0): stage_{i-1}
        # visible_start = sum(stage_len_list[:i-1]) # stage_{i-1}position
        # visible_end = sum(stage_len_list[:i+1]) # stage_iposition

        # # visible0 NEG_INF
        #     tgt_tgt_mask[row_offset:row_offset+n_rows, visible_start:visible_end] = 0.0

        #     row_offset += n_rows

        # attn_mask_2d[tgt_start:, tgt_start:] = tgt_tgt_mask

        # ---- 3. buildnotemask ablation_maskmask ----
        if ablation_mask == "causal":
            # causalmask mask
            # positionposition
            attn_mask_2d = (
                torch.triu(
                    torch.ones((L, L), device=llm_device, dtype=llm_dtype), diagonal=1
                )
                * NEG_INF
            )

        elif ablation_mask == "bidirectional":
            # bidirectionalmask visible
            # position
            attn_mask_2d = torch.zeros((L, L), device=llm_device, dtype=llm_dtype)

        else:  # ablation_mask == 'none' or 'multiscale' (default)
            # default mask mask
            attn_mask_2d = torch.zeros((L, L), device=llm_device, dtype=llm_dtype)

            # 3.1 context ↔ context: bidirectionalvisible 0

            # 3.2 context → stages: forbidden
            attn_mask_2d[:Q_context, Q_context:] = NEG_INF

            # 3.3 stages → context: 0

            # 3.4 stages ↔ stages: causalmask
            # stage_i stage_{i-1}
            tgt_start = Q_context
            tgt_tgt_mask = torch.full(
                (Q_final, Q_final),
                fill_value=NEG_INF,
                device=llm_device,
                dtype=llm_dtype,
            )

            row_offset = 0
            for i in range(len(stage_indices)):
                n_rows = stage_len_list[i]  # stage i query

                if i == 0:
                    # Stage 0:
                    visible_start = 0
                    visible_end = stage_len_list[0]
                else:
                    # Stage i (i>0): stage_{i-1}
                    visible_start = sum(stage_len_list[: i - 1])  # stage_{i-1}position
                    visible_end = sum(stage_len_list[: i + 1])  # stage_iposition

                # visible0 NEG_INF
                tgt_tgt_mask[
                    row_offset : row_offset + n_rows, visible_start:visible_end
                ] = 0.0

                row_offset += n_rows

            attn_mask_2d[tgt_start:, tgt_start:] = tgt_tgt_mask

        # ---- 4. expand4Dmergepadding mask ----
        attn_bias = (
            attn_mask_2d.unsqueeze(0).unsqueeze(1).expand(B, 1, L, L).contiguous()
        )

        # processcontextpadding mask
        if attention_mask is None:
            attention_mask = torch.ones(B, Q_context, dtype=torch.bool, device=device)
        else:
            attention_mask = attention_mask.to(device=device, dtype=torch.bool)

        # stagesdefault
        stages_mask = torch.ones(B, Q_final, dtype=torch.bool, device=device)
        mask_2d_bool = torch.cat([attention_mask, stages_mask], dim=1)  # (B, L)

        # keyvisible query
        attn_bias = attn_bias.masked_fill((~mask_2d_bool).view(B, 1, 1, L), NEG_INF)
        attn_bias = attn_bias.masked_fill((~mask_2d_bool).view(B, 1, L, 1), NEG_INF)

        # ---- 5. "" softmax NaN ----
        row_all_masked = (attn_bias <= (NEG_INF * 0.5)).all(
            dim=-1, keepdim=True
        )  # (B,1,L,1)
        eye = (
            torch.eye(L, device=llm_device, dtype=llm_dtype)
            .unsqueeze(0)
            .unsqueeze(1)
            .expand(B, 1, L, L)
        )
        fix_row = torch.full_like(attn_bias, NEG_INF)
        fix_row = torch.where(eye.bool(), torch.zeros_like(attn_bias), fix_row)
        attn_bias = torch.where(row_all_masked, fix_row, attn_bias)

        # ---- 6. numericalsanitize ----
        attn_bias = torch.nan_to_num(attn_bias, nan=0.0, posinf=POS_INF, neginf=NEG_INF)

        if use_time_position_encoding:
            # ---- 7. position_ids targets position ----
            # build Q_final [[5],[2,4],[1,3,5],...] -> [5,2,4,1,3,5,...]
            # time_index target token t
            # position_ids = [0..Q_context-1, base_offset + time_index[0], base_offset + time_index[1], ...]
            # base_offset = Q_context base position_ids position

            # 1) Python list -> Tensor = Q_final
            # note stage_indices List[List[int]]
            flat_time_index = []
            for idxs in stage_indices:
                flat_time_index.extend(idxs)  # stage
            # (Q_final,) LongTensor device
            tgt_time_index = torch.tensor(
                flat_time_index, device=device, dtype=torch.long
            )  # [Q_final]

            # 2) context position_ids
            if position_ids is None:
                # base/context position 0..Q_context-1
                base_pos = (
                    torch.arange(0, Q_context, device=device, dtype=torch.long)
                    .unsqueeze(0)
                    .repeat(B, 1)
                )  # [B, Q_context]
                # base position targets position
                # note +1 0 base 1
                base_last = base_pos[:, -1:]  # [B,1]
                pos_tgt = (
                    base_last + 1 + tgt_time_index.view(1, -1).expand(B, -1)
                )  # [B, Q_final]

                position_ids = torch.cat(
                    [base_pos, pos_tgt], dim=1
                )  # [B, Q_context + Q_final]
            else:
                # base position_ids
                # - base shape [B, Q_context] expand targets
                # - targets shape [B, Q_context + Q_final] targets part
                # targets position=base_last+1+
                assert position_ids.dim() == 2 and position_ids.size(0) == B, (
                    "Unexpected position_ids shape"
                )

                if position_ids.size(1) == Q_context:
                    base_pos = position_ids.to(
                        device=device, dtype=torch.long
                    )  # [B, Q_context]
                    base_last = base_pos[:, -1:]  # [B, 1]
                    pos_tgt = (
                        base_last + 1 + tgt_time_index.view(1, -1).expand(B, -1)
                    )  # [B, Q_final]
                    position_ids = torch.cat([base_pos, pos_tgt], dim=1)  # [B, L]
                else:
                    # targets base part targets position
                    assert position_ids.size(1) == (Q_context + Q_final), (
                        f"position_ids length must be Q_context({Q_context})+Q_final({Q_final})"
                    )
                    base_pos = position_ids[:, :Q_context].to(
                        device=device, dtype=torch.long
                    )  # [B, Q_context]
                    base_last = base_pos[:, -1:]  # [B, 1]
                    pos_tgt = (
                        base_last + 1 + tgt_time_index.view(1, -1).expand(B, -1)
                    )  # [B, Q_final]
                    position_ids = torch.cat([base_pos, pos_tgt], dim=1)
        else:
            # ---- 7. position_ids ----
            if position_ids is None:
                position_ids = (
                    torch.arange(0, L, dtype=torch.long, device=device)
                    .unsqueeze(0)
                    .repeat(B, 1)
                )
            else:
                if position_ids.size(1) == Q_context:
                    last_pos = position_ids[:, -1:]
                    incr_stages = torch.arange(
                        1, Q_final + 1, device=device, dtype=position_ids.dtype
                    ).unsqueeze(0)
                    position_ids = torch.cat(
                        [position_ids, last_pos + incr_stages], dim=1
                    )

        # ---- 8. dtype/device ----
        inputs_embeds = inputs_embeds.to(
            device=llm_device, dtype=llm_dtype
        ).contiguous()
        position_ids = position_ids.to(device=llm_device, dtype=torch.long)

        # ---- 9. mask ----
        # checkFlash Attention mask
        use_flash = (
            os.getenv("FLASHUSE", "0") == "1"
            and FLASH_ATTN_AVAILABLE
            and ablation_mask in ["causal", "bidirectional"]
            # and not return_full_hidden # flashpathreturnhidden states
        )

        if use_flash:
            # Flash Attentionpath
            logger.info(f"Using Flash Attention with mask type: {ablation_mask}")
            is_causal = ablation_mask == "causal"

            outputs = self.model.forward_with_flash_attention(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                attention_mask=mask_2d_bool,  # 2D bool mask
                is_causal=is_causal,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=return_full_hidden,
                return_dict=True,
            )
        else:
            # 4D maskpath
            if use_flash and not FLASH_ATTN_AVAILABLE:
                logger.warning(
                    "FLASHUSE=1 but Flash Attention is not available. Falling back to standard attention."
                )

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
        last_hidden = torch.nan_to_num(
            last_hidden, nan=0.0, posinf=POS_INF, neginf=NEG_INF
        )

        # stages [B, Q_final, D]
        final_tgt_output = last_hidden[:, Q_context:, :]

        if detach:
            final_tgt_output = final_tgt_output.detach()

        # return
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
            return (
                final_tgt_output,
                updated,
                outputs.hidden_states
                if outputs.hidden_states is not None
                else final_tgt_output,
            )
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
                (inputs, position_ids, attention_mask, _, inputs_embeds, _, _) = (
                    self.prepare_inputs_labels_for_multimodal_traj(
                        inputs,
                        position_ids,
                        attention_mask,
                        None,
                        None,
                        images,
                        points,
                        trajectories,
                        ego_features,
                        image_sizes=image_sizes,
                    )
                )
            else:
                (inputs, position_ids, attention_mask, _, inputs_embeds, _) = (
                    self.prepare_inputs_labels_for_multimodal(
                        inputs,
                        position_ids,
                        attention_mask,
                        None,
                        None,
                        images,
                        image_sizes=image_sizes,
                    )
                )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        inputs_embeds = sanitize_inputs_embeds(inputs_embeds)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)


def sanitize_inputs_embeds(inputs_embeds: torch.Tensor, eps: float = 1e-5):
    """
    check inputs_embeds NaN Inf
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
        print(
            f"Warning: Very large values detected in inputs_embeds (max={max_abs.item():.2e}). Clipping."
        )
        inputs_embeds = torch.clamp(inputs_embeds, min=-1e4, max=1e4)

    return inputs_embeds.to(dtype=orig_dtype)
