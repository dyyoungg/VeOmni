# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
from typing import Dict, List, Union, Optional
import json
import os

import torch
from torch.utils.data import Dataset

from ..utils import logging
from ..utils.constants import IGNORE_INDEX


logger = logging.get_logger(__name__)


class DummyTextDataset(Dataset):
    def __init__(self, size: int, seq_length: int):
        """
        Args:
            size (int): Nums of datasets
            seq_length (int, optional): seq_length
        """
        self.size = size
        self.seq_length = seq_length
        self.vocab_size = 1024

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> List[Dict[str, "torch.Tensor"]]:
        input_ids = torch.randint(low=0, high=self.vocab_size, size=(self.seq_length,))
        attention_mask = torch.ones((self.seq_length,), dtype=torch.long)
        position_ids = torch.arange(0, self.seq_length)
        labels = input_ids.clone()
        labels[0] = IGNORE_INDEX
        return [
            {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "position_ids": position_ids}
        ]


class DummyQwenVLDataset(Dataset):
    def __init__(
        self, size: int, seq_length: int, patch_size: int = 14, temporal_patch_size: int = 2, merge_size: int = 2
    ):
        """
        Args:
            size (int): Nums of datasets
            seq_length (int, optional): seq_length
        """
        self.size = size
        self.seq_length = seq_length
        self.vocab_size = 1024

        image_t = 2
        video_t = 10
        h, w = 4, 4
        channel_size = 3

        self.image_size = [image_t * h * w, patch_size * patch_size * temporal_patch_size * channel_size]
        self.image_grid_thw = torch.tensor([[1, h, w]] * image_t, dtype=torch.long)
        self.image_seqlen = h * w // (merge_size**2) * image_t

        self.video_size = [video_t * h * w, patch_size * patch_size * temporal_patch_size * channel_size]
        self.video_grid_thw = torch.tensor([[video_t, h, w]], dtype=torch.long)
        self.video_seqlen = h * w // (merge_size**2) * video_t

        self.text_seqlen = self.seq_length - self.image_seqlen - self.video_seqlen

        self.seq_length = self.text_seqlen + self.image_seqlen + self.video_seqlen
        mask = torch.zeros((self.seq_length,), dtype=torch.bool)
        self.image_mask = mask.clone()
        self.image_mask[: self.image_seqlen] = 1
        self.video_mask = mask.clone()
        self.video_mask[-self.video_seqlen :] = 1

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> List[Dict[str, "torch.Tensor"]]:
        input_ids = torch.randint(low=0, high=self.vocab_size, size=(self.seq_length,))
        attention_mask = torch.ones((self.seq_length,), dtype=torch.long)
        labels = input_ids.clone()
        labels[0] = IGNORE_INDEX
        position_ids = torch.arange(0, self.seq_length).unsqueeze(0).repeat(3, 1)
        pixel_values = torch.rand(self.image_size, dtype=torch.float32)
        pixel_values_videos = torch.rand(self.video_size, dtype=torch.float32)
        return [
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "position_ids": position_ids,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "image_mask": self.image_mask,
                "video_mask": self.video_mask,
                "image_grid_thw": self.image_grid_thw,
                "video_grid_thw": self.video_grid_thw,
            }
        ]


class DummyQwenOmniDataset(Dataset):
    def __init__(
        self, size: int, seq_length: int, patch_size: int = 14, temporal_patch_size: int = 2, merge_size: int = 2
    ):
        """
        Args:
            size (int): Nums of datasets
            seq_length (int, optional): seq_length
            dummy_data:
            [input_ids, input_image_token, input_audio_token, input_video_token, output_image_token]
        """
        self.size = size
        self.seq_length = seq_length
        self.vocab_size = 1024

        h, w = 4, 4
        image_t = 2
        video_t = 10
        channel_size = 3

        self.image_size = [image_t * h * w, patch_size * patch_size * temporal_patch_size * channel_size]
        self.image_grid_thw = torch.tensor([[1, h, w]] * image_t, dtype=torch.long)
        self.image_seqlen = h * w // (merge_size**2) * image_t

        audio_token_num = 100
        audio_num = 2
        self.audio_size = [4 * audio_token_num * audio_num, 128]
        self.audio_feature_lengths = torch.tensor([4 * audio_token_num] * audio_num, dtype=torch.long)
        self.audio_seq_length = audio_num * audio_token_num
        self.feature_attention_mask = torch.ones((audio_num, 4 * audio_token_num), dtype=torch.long)

        rest_seq_length = self.seq_length - (self.image_seqlen + self.audio_seq_length)

        self.video_size = [video_t * h * w, patch_size * patch_size * temporal_patch_size * channel_size]
        self.video_grid_thw = torch.tensor([[video_t, h, w]], dtype=torch.long)
        self.video_seqlen = h * w // (merge_size**2) * video_t

        self.text_seqlen = rest_seq_length - self.video_seqlen

        self.seq_length = self.text_seqlen + self.image_seqlen + self.audio_seq_length + self.video_seqlen
        mask = torch.zeros((self.seq_length,), dtype=torch.bool)
        start_index = self.text_seqlen
        self.image_mask = mask.clone()
        self.image_mask[start_index : start_index + self.image_seqlen] = 1
        self.audio_mask = mask.clone()
        start_index += self.image_seqlen
        self.audio_mask[start_index : start_index + self.audio_seq_length] = 1
        self.video_mask = mask.clone()
        start_index += self.audio_seq_length
        self.video_mask[start_index : start_index + self.video_seqlen] = 1

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> List[Dict[str, "torch.Tensor"]]:
        input_ids = torch.randint(low=0, high=self.vocab_size, size=(self.seq_length,))
        attention_mask = torch.ones((self.seq_length,), dtype=torch.long)
        labels = input_ids.clone()
        labels[0] = IGNORE_INDEX
        position_ids = torch.arange(0, self.seq_length).unsqueeze(0).repeat(3, 1)
        image_features = torch.rand(self.image_size, dtype=torch.float32)
        audio_features = torch.rand(self.audio_size, dtype=torch.float32)
        video_features = torch.rand(self.video_size, dtype=torch.float32)
        return [
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "position_ids": position_ids,
                "pixel_values": image_features,
                "input_features": audio_features,
                "pixel_values_videos": video_features,
                "image_mask": self.image_mask,
                "audio_mask": self.audio_mask,
                "video_mask": self.video_mask,
                "image_grid_thw": self.image_grid_thw,
                "video_grid_thw": self.video_grid_thw,
                "audio_feature_lengths": self.audio_feature_lengths,
                "feature_attention_mask": self.feature_attention_mask,
            }
        ]


class DummyQwen3OmniMoeDataset(DummyQwenOmniDataset):
    def __init__(
        self, size: int, seq_length: int, patch_size: int = 14, temporal_patch_size: int = 2, merge_size: int = 2
    ):
        super().__init__(size, seq_length, patch_size, temporal_patch_size, merge_size)
        self.audio_seq_length = self._get_feat_extract_output_lengths(self.audio_seq_length * 4)
        self.seq_length = self.text_seqlen + self.image_seqlen + self.audio_seq_length + self.video_seqlen
        mask = torch.zeros((self.seq_length,), dtype=torch.bool)
        start_index = self.text_seqlen
        self.image_mask = mask.clone()
        self.image_mask[start_index : start_index + self.image_seqlen] = 1
        self.audio_mask = mask.clone()
        start_index += self.image_seqlen
        self.audio_mask[start_index : start_index + self.audio_seq_length] = 1
        self.video_mask = mask.clone()
        start_index += self.audio_seq_length
        self.video_mask[start_index : start_index + self.video_seqlen] = 1

    def _get_feat_extract_output_lengths(self, input_lengths):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """

        input_lengths_leave = input_lengths % 100
        feat_lengths = (input_lengths_leave - 1) // 2 + 1
        output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
        return output_lengths


class DummyUGDataset(Dataset):
    def __init__(
        self, size: int, seq_length: int, patch_size: int = 14, temporal_patch_size: int = 2, merge_size: int = 2
    ):
        """
        Args:
            size (int): Nums of datasets
            seq_length (int, optional): seq_length
            dummy_data:
            [input_ids, input_image_token, input_audio_token, input_video_token, output_image_token]
        """
        self.size = size
        self.seq_length = seq_length
        self.vocab_size = 1024

        h, w = 18, 18
        image_t = 2
        video_t = 10
        channel_size = 3

        self.input_image_size = [image_t * h * w, patch_size * patch_size * temporal_patch_size * channel_size]
        self.input_image_grid_thw = torch.tensor([[1, h, w]] * image_t, dtype=torch.long)
        self.input_image_seq_length = image_t * h * w // (merge_size**2)

        audio_token_num = 100
        audio_num = 2
        self.input_audio_size = [4 * audio_token_num * audio_num, 128]
        self.input_audio_feature_lengths = torch.tensor([4 * audio_token_num] * audio_num, dtype=torch.long)
        self.input_audio_seq_length = audio_num * audio_token_num

        output_image_token_num = 1024
        output_image_num = 1
        self.output_image_size = [output_image_num, 3, 256, 256]
        self.output_image_seq_length = output_image_num * output_image_token_num

        rest_seq_length = self.seq_length - (
            self.input_image_seq_length + self.input_audio_seq_length + self.output_image_seq_length
        )

        self.input_video_size = [video_t * h * w, patch_size * patch_size * temporal_patch_size * channel_size]
        self.input_video_grid_thw = torch.tensor([[video_t, h, w]], dtype=torch.long)
        self.video_seq_length = video_t * h * w // (merge_size**2)

        self.text_seq_length = rest_seq_length - self.video_seq_length

        self.seq_length = (
            self.text_seq_length
            + self.input_image_seq_length
            + self.input_audio_seq_length
            + self.video_seq_length
            + self.output_image_seq_length
        )
        mask = torch.zeros((self.seq_length,), dtype=torch.bool)
        start_index = self.text_seq_length
        self.image_input_mask = mask.clone()
        self.image_input_mask[start_index : start_index + self.input_image_seq_length] = 1
        self.audio_input_mask = mask.clone()
        start_index += self.input_image_seq_length
        self.audio_input_mask[start_index : start_index + self.input_audio_seq_length] = 1
        self.video_input_mask = mask.clone()
        start_index += self.input_audio_seq_length
        self.video_input_mask[start_index : start_index + self.video_seq_length] = 1
        self.image_output_mask = mask.clone()
        start_index += self.video_seq_length
        self.image_output_mask[start_index : start_index + self.output_image_seq_length] = 1

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> List[Dict[str, "torch.Tensor"]]:
        input_ids = torch.randint(low=0, high=self.vocab_size, size=(self.seq_length,))
        attention_mask = torch.ones((self.seq_length,), dtype=torch.long)
        labels = input_ids.clone()
        position_ids = torch.arange(0, self.seq_length).unsqueeze(0).repeat(3, 1)
        image_input_features = torch.rand(self.input_image_size, dtype=torch.float32)
        audio_input_features = torch.rand(self.input_audio_size, dtype=torch.float32)
        video_input_features = torch.rand(self.input_video_size, dtype=torch.float32)
        image_output_features = torch.rand(self.output_image_size, dtype=torch.float32)
        return [
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "position_ids": position_ids,
                "image_input_features": image_input_features,
                "audio_input_features": audio_input_features,
                "video_input_features": video_input_features,
                "image_output_features": image_output_features,
                "image_input_mask": self.image_input_mask,
                "audio_input_mask": self.audio_input_mask,
                "video_input_mask": self.video_input_mask,
                "image_output_mask": self.image_output_mask,
                "image_input_grid_thw": self.input_image_grid_thw,
                "video_input_grid_thw": self.input_video_grid_thw,
                "audio_input_feature_lengths": self.input_audio_feature_lengths,
            }
        ]


class RealTextDataset(Dataset):
    def __init__(
        self, 
        data_path: str, 
        tokenizer, 
        max_seq_length: int = 2048,
        human_token: str = "<|im_start|>user\n",
        assistant_token: str = "<|im_start|>assistant\n",
        end_token: str = "<|im_end|>\n"
    ):
        """
        Real dataset that reads JSONL or JSON files with standard messages format.
        
        Args:
            data_path (str): Path to JSONL or JSON file containing conversation data
            tokenizer: Tokenizer to use for encoding text
            max_seq_length (int): Maximum sequence length for tokenization
            human_token (str): Token to use for human messages
            assistant_token (str): Token to use for assistant messages  
            end_token (str): Token to use for ending conversations
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.human_token = human_token
        self.assistant_token = assistant_token
        self.end_token = end_token
        
        # Load data
        self.data = self._load_data()
        logger.info(f"Loaded {len(self.data)} conversations from {data_path}")
    
    def _load_data(self) -> List[Dict]:
        """Load data from JSONL or JSON file."""
        data = []
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        if self.data_path.endswith('.jsonl'):
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping invalid JSON line: {e}")
                            continue
        elif self.data_path.endswith('.json'):
            with open(self.data_path, 'r', encoding='utf-8') as f:
                try:
                    json_data = json.load(f)
                    if isinstance(json_data, list):
                        data = json_data
                    else:
                        data = [json_data]
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON file: {e}")
        else:
            raise ValueError("File must be either .json or .jsonl format")
        
        return data
    
    def _format_conversation(self, conversation: List[Dict]) -> tuple[List[int], List[int]]:
        """Format conversation and return input_ids and labels."""
        input_ids = []
        labels = []
        
        for turn in conversation:
            # Support both "from" and "role" fields
            speaker = turn.get("from") or turn.get("role", "")
            
            if speaker == "human" or speaker == "user":
                # Human part: tokenize and add ignore index to labels
                human_text = self.human_token + turn.get("value", turn.get("content", "")) + self.end_token + self.assistant_token 
                human_tokens = self.tokenizer.encode(human_text, add_special_tokens=False)
                input_ids.extend(human_tokens)
                labels.extend([-100] * len(human_tokens))  # Ignore human tokens
                
            elif speaker == "gpt" or speaker == "assistant":
                # Assistant part: tokenize and add actual tokens to labels
                assistant_text = turn.get("value", turn.get("content", "")) + self.end_token
                assistant_tokens = self.tokenizer.encode(assistant_text, add_special_tokens=False)
                input_ids.extend(assistant_tokens)
                labels.extend(assistant_tokens)  # Include assistant tokens in loss
        
        return input_ids, labels
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> List[Dict[str, "torch.Tensor"]]:
        """Get a single conversation item."""
        item = self.data[index]
        
        # Extract conversations
        conversations = item.get("conversations", [])
        if not conversations:
            logger.warning(f"No conversations found in item {index}")
            # Return empty conversation
            conversations = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
        
        # Format conversation and get input_ids and labels
        input_ids, labels = self._format_conversation(conversations)
        
        # Truncate if too long
        if len(input_ids) > self.max_seq_length:
            input_ids = input_ids[:self.max_seq_length]
            labels = labels[:self.max_seq_length]
        
        # Convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        
        return [{
            "input_ids": input_ids,
            "attention_mask": attention_mask, 
            "labels": labels
        }]


def build_dummy_dataset(task_type: str, size: int, max_seq_len: int, data_path="", tokenizer=None) -> "Dataset":
    if task_type == "text":
        return DummyTextDataset(size=size, seq_length=max_seq_len)
    elif task_type == "qwen2vl":
        return DummyQwenVLDataset(size=size, seq_length=max_seq_len, patch_size=14)
    elif task_type == "qwen3vl":
        return DummyQwenVLDataset(size=size, seq_length=max_seq_len, patch_size=16)
    elif task_type == "qwen2omni":
        return DummyQwenOmniDataset(size=size, seq_length=max_seq_len, patch_size=14)
    elif task_type == "qwen3omni":
        return DummyQwen3OmniMoeDataset(size=size, seq_length=max_seq_len, patch_size=16)
    elif task_type == "ug":
        return DummyUGDataset(size=size, seq_length=max_seq_len)
    else:
        raise ValueError(f"Dummy dataset type ({task_type}) is not supported.")


def build_real_dataset(
    data_path: str, 
    tokenizer, 
    max_seq_length: int = 2048,
    human_token: str = "<|im_start|>user\n",
    assistant_token: str = "<|im_start|>assistant\n", 
    end_token: str = "<|im_end|>\n"
) -> "Dataset":
    """
    Build a real dataset from JSONL or JSON files.
    
    Args:
        data_path (str): Path to JSONL or JSON file containing conversation data
        tokenizer: Tokenizer to use for encoding text
        max_seq_length (int): Maximum sequence length for tokenization
        human_token (str): Token to use for human messages
        assistant_token (str): Token to use for assistant messages
        end_token (str): Token to use for ending conversations
        
    Returns:
        RealTextDataset: Dataset instance
    """
    return RealTextDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        human_token=human_token,
        assistant_token=assistant_token,
        end_token=end_token
    )
