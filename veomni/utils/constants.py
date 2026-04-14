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


IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
AUDIO_TOKEN_INDEX = -300
VIDEO_TOKEN_INDEX = -400
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_AUDIO_TOKEN = "<audio>"
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
DEFAULT_AUDIO_START_TOKEN = "<|audio_start|>"
DEFAULT_AUDIO_END_TOKEN = "<|audio_end|>"
DEFAULT_VISION_START_TOKEN = "<|vision_start|>"
DEFAULT_VISION_END_TOKEN = "<|vision_end|>"
DEFAULT_AUDIO_PAD_TOKEN = "<|audio_pad|>"

# qwenvl25
IMAGE_FACTOR = 28 
IMAGE_PACTH_SIZE = 14
IMAGE_MIN_SIDE = IMAGE_FACTOR * 4 * 1
MIN_PIXELS_SEQ = 64  # 64 token
MAX_PIXELS_SEQ = 1980 # 1980 token
MAX_RATIO = 32
MERGE_SIZE = 2
IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]

SYSTEM_PROMPTS=["You are a multimodal assistant. Understand texts, images and audios, and answer questions accurately and honestly based on the content.",
                "You are a helpful multimodal assistant. You are able to understand the visual and audio content that the user provides, and assist the user with a variety of tasks using natural language.",
                "你是一个多模态助手,你可以理解文本、图像、视频和语音，并能根据内容准确、真实地回答问题。",
                "You are an assistant that understands images、videos and audios. Use the provided input to generate accurate, relevant, and reliable responses.",
                "你是一个AI助手，擅长视觉和音频理解，请依据提供的内容进行忠实回应，不进行主观猜测或编造信息。",
                "You are a helpful AI assistant.",
                "You are a helpful AI assistant. Answer the user's questions based on the provided input content.",
                "You are an intelligent assistant capable of understanding multimodal inputs. Assist the user with their requests.",
                "You are a versatile AI assistant. Analyze the input data provided by the user and respond appropriately.",
                "You are a helpful and honest multimodal assistant. Always provide accurate information based on the content you perceive.",
                "Act as a knowledgeable assistant. Process the user's input and provide a helpful, coherent response.",
                "你是一个乐于助人的多模态AI助手。请根据用户提供的输入内容，准确地回答问题。",
                "你是一个智能助手，能够理解并处理多种形式的输入信息。请协助用户完成任务。",
                "你是一个全能型AI助手。请基于你所感知到的内容，为用户提供有帮助的回答。"
                ]
# input index
IMAGE_INPUT_INDEX = -200
VIDEO_INPUT_INDEX = -300
AUDIO_INPUT_INDEX = -400
# output index
IMAGE_OUTPUT_INDEX = -201
VIDEO_OUTPUT_INDEX = -301
AUDIO_OUTPUT_INDEX = -401


TYPE2INDEX = {
    "input": {
        "image": IMAGE_INPUT_INDEX,
        "video": VIDEO_INPUT_INDEX,
        "audio": AUDIO_INPUT_INDEX,
    },
    "output": {
        "image": IMAGE_OUTPUT_INDEX,
        "video": VIDEO_OUTPUT_INDEX,
        "audio": AUDIO_OUTPUT_INDEX,
    },
}


MODALITY = TYPE2INDEX["input"].keys() | TYPE2INDEX["output"].keys()

def get_image_video_audio_placeholder(tokenizer):
    image_token = "<|image_pad|>"
    video_token = "<|video_pad|>"
    audio_token = DEFAULT_AUDIO_PAD_TOKEN
    image_token_id = tokenizer.convert_tokens_to_ids(image_token)
    video_token_id = tokenizer.convert_tokens_to_ids(video_token)
    audio_token_id = tokenizer.convert_tokens_to_ids(audio_token)
  
    return image_token_id, video_token_id, audio_token_id


_CHAT_TEMPLATES = {
    "qwen2": dict(
        system="<|im_start|>system\n{}<|im_end|>",
        system_in_middle="\n<|im_start|>system\n{}<|im_end|>\n<|im_start|>assistant\n",
        user="\n<|im_start|>user\n{}<|im_end|>",
        assistant="\n<|im_start|>assistant\n{}<|im_end|>",
        assistant_prefix="\n<|im_start|>assistant\n",
        query_format="\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
    ),
}

_CHAT_TEMPLATES["qwen3"] = _CHAT_TEMPLATES["qwen2"]