from .constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, \
TRAJ_TOKEN_INDEX, POINT_TOKEN_INDEX, EGO_TOKEN_INDEX
from . import conversation as conversation_lib
import transformers
import torch
from typing import Dict, Optional, Sequence, List
import copy

# def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
#     prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

#     def insert_separator(X, sep):
#         return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

#     input_ids = []
#     offset = 0
#     if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
#         offset = 1
#         input_ids.append(prompt_chunks[0][0])

#     for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
#         input_ids.extend(x[offset:])

#     if return_tensors is not None:
#         if return_tensors == 'pt':
#             return torch.tensor(input_ids, dtype=torch.long)
#         raise ValueError(f'Unsupported tensor type: {return_tensors}')

def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

def tokenizer_image_traj_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, traj_token_index=TRAJ_TOKEN_INDEX, point_token_index=POINT_TOKEN_INDEX, ego_token_index=EGO_TOKEN_INDEX, return_tensors=None):
    chunks = []
    current_text = ""
    
    i = 0
    while i < len(prompt):
        if prompt[i:i+7] == '<image>':
            if current_text:
                chunks.append(('text', current_text))
                current_text = ""
            chunks.append(('image', None))
            i += 7
        elif prompt[i:i+6] == '<traj>':
            if current_text:
                chunks.append(('text', current_text))
                current_text = ""
            chunks.append(('traj', None))
            i += 6
        elif prompt[i:i+7] == '<point>':
            if current_text:
                chunks.append(('text', current_text))
                current_text = ""
            chunks.append(('point', None))
            i += 7
        elif prompt[i:i+5] == '<ego>':
            if current_text:
                chunks.append(('text', current_text))
                current_text = ""
            chunks.append(('ego', None))
            i += 5
        else:
            current_text += prompt[i]
            i += 1
    if current_text:
        chunks.append(('text', current_text))

    input_ids = []
    offset = 0
    
    # 处理BOS token
    if len(chunks) > 0 and chunks[0][0] == 'text':
        first_chunk_tokens = tokenizer(chunks[0][1]).input_ids
        if len(first_chunk_tokens) > 0 and first_chunk_tokens[0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(first_chunk_tokens[0])
            first_chunk_tokens = first_chunk_tokens[1:]
            input_ids.extend(first_chunk_tokens)
            chunks.pop(0)
    
    # 处理剩余chunks，对 image 和 traj 使用相同的插入逻辑
    for chunk_type, text in chunks:
        if chunk_type == 'image':
            input_ids.append(image_token_index)
        elif chunk_type == 'traj':
            input_ids.append(traj_token_index)
        elif chunk_type == 'point':
            input_ids.append(point_token_index)
        elif chunk_type == 'ego':
            input_ids.append(ego_token_index)
        elif chunk_type == 'text':
            chunk_tokens = tokenizer(text).input_ids
            input_ids.extend(chunk_tokens[offset:])
    
    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids
def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation

def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len

def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                # round_len = len(tokenizer_image_token(rou, tokenizer))
                # instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
                # 图文分支（见下一条对 tokenizer_image_token 的小改）
                round_len = len(tokenizer_image_token(rou, tokenizer, prepend_bos=False))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer, prepend_bos=False))
            else:
                # round_len = len(tokenizer(rou).input_ids)
                # instruction_len = len(tokenizer(parts[0]).input_ids) - 2
                # 文本分支
                round_len = len(tokenizer(rou, add_special_tokens=False).input_ids)
                instruction_len = len(tokenizer(parts[0], add_special_tokens=False).input_ids)

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    training_mode: bool =True,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        if training_mode:
            input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
        else:
            input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
            return dict(
                input_ids=input_ids,
            )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    
    input_ids = input_ids[:, :tokenizer.model_max_length]
    
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                if len(rounds) != 1:
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer_image_token(rou, tokenizer)) + len(tokenizer_image_token(conv.sep, tokenizer))
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)

def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    training_mode: bool =True,
) -> Dict:
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image, training_mode=training_mode)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)
    
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [min(len(tokenizer_image_token(prompt, tokenizer)), tokenizer.model_max_length) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt')[:tokenizer.model_max_length] for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


def preprocess_v1_traj(
    sources, # [[{'from': 'human', 'value': '<image>\nYou are driving in singapore. Here are predefined trajectories [<G0> <point>  <G1> <point>  <G2> <point>  <G3> <point>  <G4> <point>  <G5> <point>] for the ego car. Please select the best trajectory in the current scenario.'}, {'from': 'gpt', 'value': 'The best trajectory is 0.'}, {'from': 'human', 'value': 'With the selected trajectory as a reference <traj>, please provide the planning trajectory for the ego car, which has a velocity of (0.00,0.00) m/s and an acceleration of (-0.02,-0.21) m/s^2.'}, {'from': 'gpt', 'value': 'The result is [PT, (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (+0.03, 0.0)].'}]]
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    training_mode: bool =True,
    has_traj: bool = False,
) -> Dict: # 这里还是对单个batch进行连续处理，不必考虑batch size
    import os, ipdb
    if os.getenv("DATA_DEBUG") == "1": ipdb.set_trace()
    conv = conversation_lib.default_conversation.copy() # Conversation(system="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.", roles=('USER', 'ASSISTANT'), messages=[['USER', '<image>\nYou are driving in singapore. Here are predefined trajectories [<G0> <point>  <G1> <point>  <G2> <point>  <G3> <point>  <G4> <point>  <G5> <point>] for the ego car. Please select the best trajectory in the current scenario.'], ['ASSISTANT', 'The best trajectory is 0.'], ['USER', 'With the selected trajectory as a reference <traj>, please provide the planning trajectory for the ego car, which has a velocity of (0.00,0.00) m/s and an acceleration of (-0.02,-0.21) m/s^2.'], ['ASSISTANT', 'The result is [PT, (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (+0.03, 0.0)].']], offset=0, sep_style=<SeparatorStyle.TWO: 2>, sep=' ', sep2='</s>', version='v1', skip_next=False)
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]} # {'human': 'USER', 'gpt': 'ASSISTANT'}
    # Apply prompt templates
    # import pdb; pdb.set_trace()
    conversations = []
    for i, source in enumerate(sources): 
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source): # 两轮对话 len=4 每个字典里有 from value 两个key
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    # 自己改其实只需要添加一份判定完全可以直接设置conversation，但是还是要注意batch size为2
    if has_image:
        if training_mode:
            if has_traj: # 直接就是全文了 prompt "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\nYou are driving in singapore. Here are predefined trajectories [<G0> <point>  <G1> <point>  <G2> <point>  <G3> <point>  <G4> <point>  <G5> <point>] for the ego car. Please select the best trajectory in the current scenario. ASSISTANT: The best trajectory is 0.</s>USER: With the selected trajectory as a reference <traj>, please provide the planning trajectory for the ego car, which has a velocity of (0.00,0.00) m/s and an acceleration of (-0.02,-0.21) m/s^2. ASSISTANT: The result is [PT, (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (+0.03, 0.0)].</s>"
                input_ids = torch.stack([tokenizer_image_traj_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
            else:
                input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)

        else:
            if has_traj:
                input_ids = [tokenizer_image_traj_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
            else:
                input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]

            return dict(
                input_ids=input_ids,
            )

    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    
    input_ids = input_ids[:, :tokenizer.model_max_length]
    
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                if has_traj:
                    round_len = len(tokenizer_image_traj_token(rou, tokenizer))
                    instruction_len = len(tokenizer_image_traj_token(parts[0], tokenizer)) - 2
                else:
                    round_len = len(tokenizer_image_token(rou, tokenizer))
                    instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                if len(rounds) != 1:
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    training_mode: bool =True,
) -> Dict:
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image, training_mode=training_mode)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)
    
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [min(len(tokenizer_image_token(prompt, tokenizer)), tokenizer.model_max_length) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt')[:tokenizer.model_max_length] for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)

def preprocess_traj(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    training_mode: bool =True,
    has_traj: bool = False,
    use_qwen: bool = False,
    use_qwenvl_25: bool = False,
) -> Dict:
    # import os, ipdb
    # if os.getenv("DEBUG") == "1": ipdb.set_trace()
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"): # True
        return preprocess_v1_traj(sources, tokenizer, has_image=has_image, training_mode=training_mode, has_traj=has_traj)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)
    
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [min(len(tokenizer_image_token(prompt, tokenizer)), tokenizer.model_max_length) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt')[:tokenizer.model_max_length] for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)