# coding=utf-8
# Copyright 2024 The ggml.ai team and The HuggingFace Inc. team. and pygguf author (github.com/99991)
# https://github.com/99991/pygguf
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

from array import array
import numpy as np
from tqdm import tqdm

# Listed here: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
GGML_TYPES = {
    "F32": 0,
    "Q4_0": 2,
    "Q8_0": 8,
    "Q2_K": 10,
    "Q3_K": 11,
    "Q4_K": 12,
    "Q5_K": 13,
    "Q6_K": 14,
}

# The Blocksizes are reported in bytes
# Check out: https://github.com/ggerganov/llama.cpp/blob/8a56075b07a8b571bf95a912ffdce4c928c2b414/gguf-py/gguf/constants.py#L801
GGML_BLOCK_SIZES = {
    "Q8_0": 2 + 32,  # Q8_0 uses a blocksize of 32 (int8 tensors) + 2 bytes allocated for the scales
    "Q4_K": 144,
    # Q4_0 uses a blocksize of 32 but the 4-bit tensors are packed into 8-bit tensors + 2 bytes for the scales
    "Q4_0": 2 + 16,
    "Q6_K": 210,
    # See: https://github.com/99991/pygguf/commit/a417edbfc029a1bc270f984a694f9128c5afa8b9
    "Q2_K": 256 // 16 + 256 // 4 + 2 + 2,
    "Q3_K": 256 // 8 + 256 // 4 + 12 + 2,
    "Q5_K": 2 + 2 + 12 + 256 // 8 + 256 // 2,
}

# Listed here: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
DATA_TYPES = {
    "uint32": 4,
    "int32": 5,
    "float32": 6,
    "bool": 7,
    "string": 8,
    "array": 9,
    "uint64": 10,
}


def dequantize_q8_0(data):
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L43
    block_size = GGML_BLOCK_SIZES["Q8_0"]
    num_blocks = len(data) // block_size

    scales = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, 1 + 16)[:, :1].astype(np.float32)
    qs = np.frombuffer(data, dtype=np.int8).reshape(num_blocks, 2 + 32)[:, 2:]

    return scales * qs

def dequantize_q4_k(data):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L1929
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L116
    block_size = GGML_BLOCK_SIZES["Q4_K"]
    num_blocks = len(data) // block_size

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, block_size // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, block_size)

    # Casting to float32 because float16 is very slow on CPU
    scale_factors = data_f16[:, 0].reshape(num_blocks, 1, 1).astype(np.float32)
    scale_offsets = data_f16[:, 1].reshape(num_blocks, 1, 1).astype(np.float32)
    qs1 = data_u8[:, 4:16].reshape(num_blocks, 12, 1)
    qs2 = data_u8[:, 16:].reshape(num_blocks, 4, 32)

    # Dequantize scales and offsets (6 bits and 4 + 2 bits)
    factors = scale_factors * np.concatenate(
        [qs1[:, 0:4] & 0b111111, (qs1[:, 8:] & 15) | ((qs1[:, 0:4] >> 6) << 4)], axis=1
    )
    offsets = scale_offsets * np.concatenate(
        [qs1[:, 4:8] & 0b111111, (qs1[:, 8:] >> 4) | ((qs1[:, 4:8] >> 6) << 4)], axis=1
    )

    # Interleave low and high quantized bits
    qs2 = np.stack([qs2 & 0xF, qs2 >> 4], axis=2).reshape(num_blocks, 8, 32)
    # Dequantize final weights using scales and offsets
    return factors * qs2 - offsets


def dequantize_q4_0(data):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L1086
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L11
    block_size = GGML_BLOCK_SIZES["Q4_0"]
    num_blocks = len(data) // block_size

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, block_size // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, block_size)

    # The scales are stored on the first 2 bytes and the rest corresponds to the quants
    scales = data_f16[:, 0].reshape(num_blocks, 1).astype(np.float32)
    # scales = np.nan_to_num(scales)
    # the rest of the bytes corresponds to the quants - we discard the first two bytes
    quants = data_u8[:, 2:]

    ql = (quants[:, :] & 0xF).astype(np.int8) - 8
    qr = (quants[:, :] >> 4).astype(np.int8) - 8

    # Use hstack
    quants = np.hstack([ql, qr])

    return (scales * quants).astype(np.float32)

def dequantize_q6_k(data):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L2275
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L152
    block_size = GGML_BLOCK_SIZES["Q6_K"]
    num_blocks = len(data) // block_size

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, block_size // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, block_size)
    data_i8 = np.frombuffer(data, dtype=np.int8).reshape(num_blocks, block_size)

    scales = data_f16[:, -1].reshape(num_blocks, 1).astype(np.float32)

    # TODO use uint8 and cast later?
    ql = data_u8[:, :128].astype(np.int16)
    qh = data_u8[:, 128:192].astype(np.int16)
    sc = data_i8[:, 192:208, np.newaxis].astype(np.float32)

    # Unpack bits, subtraction requires signed data type
    q1 = (ql[:, :32] & 0xF) | (((qh[:, :32] >> 0) & 3) << 4) - 32
    q2 = (ql[:, 32:64] & 0xF) | (((qh[:, :32] >> 2) & 3) << 4) - 32
    q3 = (ql[:, :32] >> 4) | (((qh[:, :32] >> 4) & 3) << 4) - 32
    q4 = (ql[:, 32:64] >> 4) | (((qh[:, :32] >> 6) & 3) << 4) - 32
    q5 = (ql[:, 64:96] & 0xF) | (((qh[:, 32:] >> 0) & 3) << 4) - 32
    q6 = (ql[:, 96:128] & 0xF) | (((qh[:, 32:] >> 2) & 3) << 4) - 32
    q7 = (ql[:, 64:96] >> 4) | (((qh[:, 32:] >> 4) & 3) << 4) - 32
    q8 = (ql[:, 96:128] >> 4) | (((qh[:, 32:] >> 6) & 3) << 4) - 32

    # Dequantize
    return scales * np.concatenate(
        [
            sc[:, 0] * q1[:, :16],
            sc[:, 1] * q1[:, 16:],
            sc[:, 2] * q2[:, :16],
            sc[:, 3] * q2[:, 16:],
            sc[:, 4] * q3[:, :16],
            sc[:, 5] * q3[:, 16:],
            sc[:, 6] * q4[:, :16],
            sc[:, 7] * q4[:, 16:],
            sc[:, 8] * q5[:, :16],
            sc[:, 9] * q5[:, 16:],
            sc[:, 10] * q6[:, :16],
            sc[:, 11] * q6[:, 16:],
            sc[:, 12] * q7[:, :16],
            sc[:, 13] * q7[:, 16:],
            sc[:, 14] * q8[:, :16],
            sc[:, 15] * q8[:, 16:],
        ],
        axis=1,
    )

def dequantize_q2_k(data):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L1547
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L74
    num_blocks = len(data) // GGML_BLOCK_SIZES["Q2_K"]

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, GGML_BLOCK_SIZES["Q2_K"] // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, GGML_BLOCK_SIZES["Q2_K"])

    dmin = data_f16[:, -1].reshape(num_blocks, 1, 1).astype(np.float32)
    d = data_f16[:, -2].reshape(num_blocks, 1, 1).astype(np.float32)
    scales = data_u8[:, :16].reshape(num_blocks, 16, 1)
    qs = data_u8[:, 16:80].reshape(num_blocks, 64)

    tmp = np.stack(
        [
            qs[:, 00:16] >> 0,
            qs[:, 16:32] >> 0,
            qs[:, 00:16] >> 2,
            qs[:, 16:32] >> 2,
            qs[:, 00:16] >> 4,
            qs[:, 16:32] >> 4,
            qs[:, 00:16] >> 6,
            qs[:, 16:32] >> 6,
            qs[:, 32:48] >> 0,
            qs[:, 48:64] >> 0,
            qs[:, 32:48] >> 2,
            qs[:, 48:64] >> 2,
            qs[:, 32:48] >> 4,
            qs[:, 48:64] >> 4,
            qs[:, 32:48] >> 6,
            qs[:, 48:64] >> 6,
        ],
        axis=1,
    )

    return d * (scales & 15) * (tmp & 3) - dmin * (scales >> 4)

def dequantize_q3_k(data):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L1723C32-L1723C42
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L95
    num_blocks = len(data) // GGML_BLOCK_SIZES["Q3_K"]

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, GGML_BLOCK_SIZES["Q3_K"] // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, GGML_BLOCK_SIZES["Q3_K"])

    d = data_f16[:, -1].reshape(num_blocks, 1, 1).astype(np.float32)
    bits = np.unpackbits(data_u8[:, :32].reshape(num_blocks, 32, 1), axis=-1, bitorder="little")
    bits = 4 ^ (bits << 2)
    qs = data_u8[:, 32 : 32 + 64].astype(np.int16)
    a, b, c = data_u8[:, 96 : 96 + 12].reshape(num_blocks, 3, 4).transpose(1, 0, 2)
    scales = np.zeros((num_blocks, 4, 4), dtype=np.uint8)
    scales[:, 0] = (a & 15) | ((c & 3) << 4)
    scales[:, 1] = (b & 15) | (((c >> 2) & 3) << 4)
    scales[:, 2] = (a >> 4) | (((c >> 4) & 3) << 4)
    scales[:, 3] = (b >> 4) | ((c >> 6) << 4)
    scales = scales.reshape(num_blocks, 16, 1).astype(np.int16)

    return (
        d
        * (scales - 32)
        * np.stack(
            [
                (((qs[:, 00:16] >> 0) & 3) - bits[:, :16, 0]),
                (((qs[:, 16:32] >> 0) & 3) - bits[:, 16:, 0]),
                (((qs[:, 00:16] >> 2) & 3) - bits[:, :16, 1]),
                (((qs[:, 16:32] >> 2) & 3) - bits[:, 16:, 1]),
                (((qs[:, 00:16] >> 4) & 3) - bits[:, :16, 2]),
                (((qs[:, 16:32] >> 4) & 3) - bits[:, 16:, 2]),
                (((qs[:, 00:16] >> 6) & 3) - bits[:, :16, 3]),
                (((qs[:, 16:32] >> 6) & 3) - bits[:, 16:, 3]),
                (((qs[:, 32:48] >> 0) & 3) - bits[:, :16, 4]),
                (((qs[:, 48:64] >> 0) & 3) - bits[:, 16:, 4]),
                (((qs[:, 32:48] >> 2) & 3) - bits[:, :16, 5]),
                (((qs[:, 48:64] >> 2) & 3) - bits[:, 16:, 5]),
                (((qs[:, 32:48] >> 4) & 3) - bits[:, :16, 6]),
                (((qs[:, 48:64] >> 4) & 3) - bits[:, 16:, 6]),
                (((qs[:, 32:48] >> 6) & 3) - bits[:, :16, 7]),
                (((qs[:, 48:64] >> 6) & 3) - bits[:, 16:, 7]),
            ],
            axis=1,
        )
    )


def dequantize_q5_k(data):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L2129
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L138
    num_blocks = len(data) // GGML_BLOCK_SIZES["Q5_K"]

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, GGML_BLOCK_SIZES["Q5_K"] // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, GGML_BLOCK_SIZES["Q5_K"])

    d = data_f16[:, 0].reshape(num_blocks, 1).astype(np.float32)
    dmin = data_f16[:, 1].reshape(num_blocks, 1).astype(np.float32)
    scales = data_u8[:, 4:16].reshape(num_blocks, 12, 1)
    qh = data_u8[:, 16 : 16 + 32].reshape(num_blocks, 32, 1)
    qs = data_u8[:, 48 : 48 + 128].reshape(num_blocks, 4, 32)

    bits = np.unpackbits(qh, axis=-1, bitorder="little")

    qs_hi_4 = qs >> 4
    qs_lo_4 = qs & 15

    scales_lo_6 = scales[:, :8] & 63
    scales_hi_6 = scales[:, :8] >> 6
    scales_lo_4 = scales[:, 8:] & 15
    scales_hi_4 = scales[:, 8:] >> 4

    m1 = dmin * scales_lo_6[:, 4]
    m2 = dmin * scales_lo_6[:, 5]
    m3 = dmin * scales_lo_6[:, 6]
    m4 = dmin * scales_lo_6[:, 7]
    m5 = dmin * (scales_hi_4[:, 0] | (scales_hi_6[:, 4] << 4))
    m6 = dmin * (scales_hi_4[:, 1] | (scales_hi_6[:, 5] << 4))
    m7 = dmin * (scales_hi_4[:, 2] | (scales_hi_6[:, 6] << 4))
    m8 = dmin * (scales_hi_4[:, 3] | (scales_hi_6[:, 7] << 4))

    d1 = d * scales_lo_6[:, 0]
    d2 = d * scales_lo_6[:, 1]
    d3 = d * scales_lo_6[:, 2]
    d4 = d * scales_lo_6[:, 3]
    d5 = d * (scales_lo_4[:, 0] | (scales_hi_6[:, 0] << 4))
    d6 = d * (scales_lo_4[:, 1] | (scales_hi_6[:, 1] << 4))
    d7 = d * (scales_lo_4[:, 2] | (scales_hi_6[:, 2] << 4))
    d8 = d * (scales_lo_4[:, 3] | (scales_hi_6[:, 3] << 4))

    return np.concatenate(
        [
            d1 * (qs_lo_4[:, 0] + (bits[:, :, 0] << 4)) - m1,
            d2 * (qs_hi_4[:, 0] + (bits[:, :, 1] << 4)) - m2,
            d3 * (qs_lo_4[:, 1] + (bits[:, :, 2] << 4)) - m3,
            d4 * (qs_hi_4[:, 1] + (bits[:, :, 3] << 4)) - m4,
            d5 * (qs_lo_4[:, 2] + (bits[:, :, 4] << 4)) - m5,
            d6 * (qs_hi_4[:, 2] + (bits[:, :, 5] << 4)) - m6,
            d7 * (qs_lo_4[:, 3] + (bits[:, :, 6] << 4)) - m7,
            d8 * (qs_hi_4[:, 3] + (bits[:, :, 7] << 4)) - m8,
        ],
        axis=1,
    )

def load_dequant_gguf_tensor(shape, ggml_type, data):
    if ggml_type == GGML_TYPES["F32"]:
        values = data
    elif ggml_type == GGML_TYPES["Q8_0"]:
        values = dequantize_q8_0(data)
    elif ggml_type == GGML_TYPES["Q4_0"]:
        values = dequantize_q4_0(data)
    elif ggml_type == GGML_TYPES["Q4_K"]:
        values = dequantize_q4_k(data)
    elif ggml_type == GGML_TYPES["Q6_K"]:
        values = dequantize_q6_k(data)
    elif ggml_type == GGML_TYPES["Q2_K"]:
        values = dequantize_q2_k(data)
    elif ggml_type == GGML_TYPES["Q3_K"]:
        values = dequantize_q3_k(data)
    elif ggml_type == GGML_TYPES["Q5_K"]:
        values = dequantize_q5_k(data)
    else:
        raise NotImplementedError(
            f"ggml_type {ggml_type} not implemented - please raise an issue on huggingface transformers: https://github.com/huggingface/transformers/issues/new/choose"
        )

    return values.reshape(shape[::-1])

def _gguf_parse_value(_value, data_type):
    if not isinstance(data_type, list):
        data_type = [data_type]
    if len(data_type) == 1:
        data_type = data_type[0]
        array_data_type = None
    else:
        if data_type[0] != 9:
            raise ValueError("Received multiple types, therefore expected the first type to indicate an array.")
        data_type, array_data_type = data_type

    if data_type in [0, 1, 2, 3, 4, 5, 10, 11]:
        _value = int(_value[0])
    elif data_type in [6, 12]:
        _value = float(_value[0])
    elif data_type in [7]:
        _value = bool(_value[0])
    elif data_type in [8]:
        _value = array("B", list(_value)).tobytes().decode()
    elif data_type in [9]:
        _value = _gguf_parse_value(_value, array_data_type)
    return _value

GGUF_TOKENIZER_MAPPING = {
    "tokenizer": {
        "ggml.model": "tokenizer_type",
        "ggml.tokens": "tokens",
        "ggml.scores": "scores",
        "ggml.token_type": "token_type",
        "ggml.merges": "merges",
        "ggml.bos_token_id": "bos_token_id",
        "ggml.eos_token_id": "eos_token_id",
        "ggml.unknown_token_id": "unk_token_id",
        "ggml.padding_token_id": "pad_token_id",
        "ggml.add_space_prefix": "add_prefix_space",
    },
    "tokenizer_config": {
        "chat_template": "chat_template",
        "ggml.model": "model_type",
        "ggml.bos_token_id": "bos_token_id",
        "ggml.eos_token_id": "eos_token_id",
        "ggml.unknown_token_id": "unk_token_id",
        "ggml.padding_token_id": "pad_token_id",
    },
}

GGUF_TENSOR_MAPPING = {
    "llama": {
        "token_embd": "model.embed_tokens",
        "blk": "model.layers",
        "ffn_up": "mlp.up_proj",
        "ffn_down": "mlp.down_proj",
        "ffn_gate": "mlp.gate_proj",
        "ffn_norm": "post_attention_layernorm",
        "attn_norm": "input_layernorm",
        "attn_q": "self_attn.q_proj",
        "attn_v": "self_attn.v_proj",
        "attn_k": "self_attn.k_proj",
        "attn_output": "self_attn.o_proj",
        "output.weight": "lm_head.weight",
        "output_norm": "model.norm",
    },
    "mistral": {
        "token_embd": "model.embed_tokens",
        "blk": "model.layers",
        "ffn_up": "mlp.up_proj",
        "ffn_down": "mlp.down_proj",
        "ffn_gate": "mlp.gate_proj",
        "ffn_norm": "post_attention_layernorm",
        "attn_norm": "input_layernorm",
        "attn_q": "self_attn.q_proj",
        "attn_v": "self_attn.v_proj",
        "attn_k": "self_attn.k_proj",
        "attn_output": "self_attn.o_proj",
        "output.weight": "lm_head.weight",
        "output_norm": "model.norm",
    },
    "qwen2": {
        "token_embd": "model.embed_tokens",
        "blk": "model.layers",
        "ffn_up": "mlp.up_proj",
        "ffn_down": "mlp.down_proj",
        "ffn_gate": "mlp.gate_proj",
        "ffn_norm": "post_attention_layernorm",
        "attn_norm": "input_layernorm",
        "attn_q": "self_attn.q_proj",
        "attn_v": "self_attn.v_proj",
        "attn_k": "self_attn.k_proj",
        "attn_output": "self_attn.o_proj",
        "output.weight": "lm_head.weight",
        "output_norm": "model.norm",
    },
}


GGUF_CONFIG_MAPPING = {
    "general": {
        "architecture": "model_type",
        "name": "_model_name_or_path",
    },
    "llama": {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
    },
    "mistral": {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
    },
    "qwen2": {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
    },
    "tokenizer": {
        "ggml.bos_token_id": "bos_token_id",
        "ggml.eos_token_id": "eos_token_id",
        "ggml.unknown_token_id": "unk_token_id",
        "ggml.padding_token_id": "pad_token_id",
    },
}

from .logging import get_logger



import torch

logger = get_logger(__name__)


GGUF_TO_TRANSFORMERS_MAPPING = {
    "ignore": {
        "GGUF": {
            "version": "version",
            "tensor_count": "tensor_count",
            "kv_count": "kv_count",
        },
        "general": {"file_type": "file_type", "quantization_version": "quantization_version"},
    },
    "config": GGUF_CONFIG_MAPPING,
    "tensors": GGUF_TENSOR_MAPPING,
    "tokenizer": {"tokenizer": GGUF_TOKENIZER_MAPPING["tokenizer"]},
    "tokenizer_config": {"tokenizer": GGUF_TOKENIZER_MAPPING["tokenizer_config"]},
}

GGUF_SUPPORTED_ARCHITECTURES = list(GGUF_TO_TRANSFORMERS_MAPPING["tensors"].keys())


def read_field(reader, field):
    value = reader.fields[field]
    return [_gguf_parse_value(value.parts[_data_index], value.types) for _data_index in value.data]


def load_gguf_checkpoint(gguf_checkpoint_path, return_tensors=False):
    """
    Load a GGUF file and return a dictionary of parsed parameters containing tensors, the parsed
    tokenizer and config attributes.

    Args:
        gguf_checkpoint_path (`str`):
            The path the to GGUF file to load
        return_tensors (`bool`, defaults to `True`):
            Whether to read the tensors from the file and return them. Not doing so is faster
            and only loads the metadata in memory.
    """
    try:
        from gguf import GGUFReader
    except (ImportError, ModuleNotFoundError):
        logger.error(
            "Loading a GGUF checkpoint in PyTorch, requires both PyTorch and GGUF to be installed. Please see "
            "https://pytorch.org/ and https://github.com/ggerganov/llama.cpp/tree/master/gguf-py for installation instructions."
        )
        raise

    reader = GGUFReader(gguf_checkpoint_path)
    fields = reader.fields
    reader_keys = list(fields.keys())

    parsed_parameters = {k: {} for k in GGUF_TO_TRANSFORMERS_MAPPING}

    architecture = read_field(reader, "general.architecture")[0]
    model_name = read_field(reader, "general.name")

    # in llama.cpp mistral models use the same architecture as llama. We need
    # to add this patch to ensure things work correctly on our side.
    if "llama" in architecture and "mistral" in model_name:
        updated_architecture = "mistral"
    else:
        updated_architecture = architecture

    if architecture not in GGUF_SUPPORTED_ARCHITECTURES:
        raise ValueError(f"Architecture {architecture} not supported")

    # List all key-value pairs in a columnized format
    for gguf_key, field in reader.fields.items():
        gguf_key = gguf_key.replace(architecture, updated_architecture)
        split = gguf_key.split(".")
        prefix = split[0]
        config_key = ".".join(split[1:])

        value = [_gguf_parse_value(field.parts[_data_index], field.types) for _data_index in field.data]

        if len(value) == 1:
            value = value[0]

        if isinstance(value, str) and architecture in value:
            value = value.replace(architecture, updated_architecture)

        for parameter in GGUF_TO_TRANSFORMERS_MAPPING:
            parameter_renames = GGUF_TO_TRANSFORMERS_MAPPING[parameter]
            if prefix in parameter_renames and config_key in parameter_renames[prefix]:
                renamed_config_key = parameter_renames[prefix][config_key]
                if renamed_config_key == -1:
                    continue

                if renamed_config_key is not None:
                    parsed_parameters[parameter][renamed_config_key] = value

                if gguf_key in reader_keys:
                    reader_keys.remove(gguf_key)

        if gguf_key in reader_keys:
            logger.info(f"Some keys were not parsed and added into account {gguf_key} | {value}")

    if return_tensors:
        tensor_key_mapping = GGUF_TO_TRANSFORMERS_MAPPING["tensors"][architecture]

        for tensor in tqdm(reader.tensors, desc="Converting and de-quantizing GGUF tensors..."):
            renamed_tensor_name = tensor.name

            for tensor_name_mapping in GGUF_TO_TRANSFORMERS_MAPPING["tensors"]:
                if tensor_name_mapping in renamed_tensor_name:
                    renamed_tensor_name = renamed_tensor_name.replace(
                        tensor_name_mapping, GGUF_TO_TRANSFORMERS_MAPPING["tensors"][tensor_name_mapping]
                    )

            shape = tensor.shape
            name = tensor.name

            weights = load_dequant_gguf_tensor(shape=shape, ggml_type=tensor.tensor_type, data=tensor.data)

            if architecture == "llama" and (".attn_k." in name or ".attn_q." in name):
                num_heads = parsed_parameters["config"]["num_attention_heads"]
                tmp_shape = (int(shape[-1] // num_heads // 2), num_heads, 2, shape[0])
                weights = weights.reshape(tmp_shape)
                weights = weights.transpose(0, 2, 1, 3)
                weights = weights.reshape(shape[::-1])

            for tensor_name in tensor_key_mapping:
                if tensor_name in name:
                    name = name.replace(tensor_name, tensor_key_mapping[tensor_name])

            # Use copy to avoid errors with numpy and pytorch
            parsed_parameters["tensors"][name] = torch.from_numpy(np.copy(weights))

    if len(reader_keys) > 0:
        logger.info(f"Some keys of the GGUF file were not considered: {reader_keys}")

    return parsed_parameters