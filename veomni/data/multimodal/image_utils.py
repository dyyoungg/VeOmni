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


import io
import math
from io import BytesIO
from typing import ByteString, List, Union
from functools import lru_cache

import numpy as np
import requests
from PIL import Image
import base64
from typing import List, Union, Optional, Tuple
import requests
from PIL.Image import Resampling
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import StoppingCriteria
from veomni.utils.constants import IMAGE_TOKEN_INDEX, AUDIO_TOKEN_INDEX, IMAGE_FACTOR, MIN_PIXELS, MAX_PIXELS, MAX_RATIO, IMAGE_MEAN, IMAGE_MIN_SIDE
from concurrent.futures import ThreadPoolExecutor
from transformers import Qwen2_5_VLProcessor
from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessorKwargs
from transformers.feature_extraction_utils import BatchFeature


ImageInput = Union[
    Image.Image,
    np.ndarray,
    ByteString,
    str,
]


def load_image_bytes_from_path(image_path: str):
    image = Image.open(image_path).convert("RGB")
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    return image_bytes.getvalue()


def save_image_bytes_to_file(image_bytes, output_path):
    image_bytes = io.BytesIO(image_bytes)
    image = Image.open(image_bytes).convert("RGB")
    image.save(output_path)


def smart_resize(
    height: int, 
    width: int, 
    factor: int = IMAGE_FACTOR, 
    min_pixels: int = MIN_PIXELS, 
    max_pixels: int = MAX_PIXELS,
    min_side: int = IMAGE_MIN_SIDE
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    # possible_resolution = [[644,364], [336,336], [448, 448], [364, 644], [560, 168], [168, 560]]
    # width, height = select_best_resolution((width, height), possible_resolution)
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)

    if min_side > 0:
        
        if h_bar < min_side:
            beta = min_side / float(h_bar)
            h_bar = ceil_by_factor(min_side, factor)
            w_bar = ceil_by_factor(w_bar * beta, factor)
        
        if w_bar < min_side:
            beta = min_side / float(w_bar)
            w_bar = ceil_by_factor(min_side, factor)
            h_bar = ceil_by_factor(h_bar * beta, factor)
    
    return h_bar, w_bar


def load_image_from_path(image: str, **kwargs):
    if image.startswith("http://") or image.startswith("https://"):
        response = requests.get(image, stream=True)
        image_obj = Image.open(BytesIO(response.content))
    else:
        image_obj = Image.open(image)
    return image_obj.convert("RGB")


def load_image_from_bytes(image: bytes, **kwargs):
    return Image.open(BytesIO(image)).convert("RGB")


def load_image(image: ImageInput, **kwargs):
    if isinstance(image, str):
        return load_image_from_path(image, **kwargs)
    elif isinstance(image, bytes):
        return load_image_from_bytes(image, **kwargs)
    else:
        raise NotImplementedError


def fetch_images(images: List[ImageInput], **kwargs):
    images = [load_image(image) for image in images]
    max_image_nums = kwargs.get("max_image_nums", len(images))
    images = images[:max_image_nums]
    images = [smart_resize(image, **kwargs) for image in images]
    return images



def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, model_cfg, **kwargs):
    batch = []
    for image in images:
        image_width, image_height = image.size
        if image_width * image_processor.size['height'] < image_height * image_processor.size['width']:
            padded_width = round(image_height * image_processor.size['width'] / image_processor.size['height'])
            padded_height = image_height
        else:
            padded_height = round(image_width * image_processor.size['height'] / image_processor.size['width'])
            padded_width = image_width
        bgcolor = tuple(int(x * 255) for x in image_processor.image_mean)
        padded_image = Image.new(image.mode, (padded_width, padded_height), bgcolor)
        padded_image.paste(image, ((padded_width - image_width) // 2, (padded_height - image_height) // 2))
        padded_image = padded_image.resize((image_processor.size['width'], image_processor.size['height']), image_processor.resample)
        batch.append(padded_image)
    
    kw = dict(return_tensors="pt")
    kw.update(kwargs)
    new_images = image_processor.preprocess(
        batch, do_resize=False, **kw
    )["pixel_values"]

    # if all(x.shape == new_images[0].shape for x in new_images):
    #     new_images = torch.stack(new_images, dim=0)
    return new_images


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    try:
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]
    except:
        print('error')
        print(prompt)
        

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


def tokenizer_audio_token(prompt, tokenizer, audio_token_index=AUDIO_TOKEN_INDEX, return_tensors=None):
    try:
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<audio>')]
    except:
        print('error')
        print(prompt)
        

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])
    
    for x in insert_separator(prompt_chunks, [audio_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids



def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]




class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False


def to_rgb(pil_image: Image.Image) -> Image.Image:
    if pil_image.mode == 'RGBA':
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
        return white_background
    if pil_image.mode == 'RGB':
        return pil_image
    else:
        return pil_image.convert("RGB")
    
def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float('inf')

    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit

def pad_image(image, h, w):
    image_width, image_height = image.size
    if image_width * h < image_height * w:
        padded_width = round(image_height * w / h)
        padded_height = image_height
    else:
        padded_height = round(image_width * h / w)
        padded_width = image_width
    bgcolor = tuple(int(x * 255) for x in IMAGE_MEAN)
    padded_image = Image.new(image.mode, (padded_width, padded_height), bgcolor)
    padded_image.paste(image, ((padded_width - image_width) // 2, (padded_height - image_height) // 2))
    padded_image = padded_image.resize((w, h), Resampling.BILINEAR)
    return padded_image

def _process_one_image(image_input, min_side, size_factor, force_fixed_size=None):
    """
    Args:
        force_fixed_size: Optional[Tuple[int, int]], e.g., (336, 336). 
                          If provided, ignores smart_resize logic and forces resize.
    """
    image_obj = None
    try:
        if isinstance(image_input, Image.Image):
            image_obj = image_input
        elif isinstance(image_input, str):
            if image_input.startswith("http://") or image_input.startswith("https://"):
                response = requests.get(image_input, stream=True)
                image_obj = Image.open(BytesIO(response.content))
            elif image_input.startswith("file://"):
                image_obj = Image.open(image_input[7:])
            elif image_input.startswith("data:image"):
                if "base64," in image_input:
                    _, base64_data = image_input.split("base64,", 1)
                    data = base64.b64decode(base64_data)
                    image_obj = Image.open(BytesIO(data))
            else: 
                image_obj = Image.open(image_input)
        
        if image_obj is None:
            raise ValueError(f"Unrecognized image input: {image_input}")

        image = to_rgb(image_obj)
        width, height = image.size

      
        if force_fixed_size is not None:
           
            target_w, target_h = force_fixed_size
            divided_factor = IMAGE_FACTOR * 4
            if target_w % divided_factor != 0 or target_h % divided_factor != 0:
                print(f"[Warning] force_fixed_size {force_fixed_size} is not divisible by {divided_factor}! "
                      f"Might cause shape mismatch in Conv Projector.")
                
            if (width, height) != (target_w, target_h):
                image = image.resize((target_w, target_h), resample=Image.BICUBIC)
            
            resized_width, resized_height = target_w, target_h
            
        else:
         
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=size_factor * 4,  
                min_pixels=MIN_PIXELS, 
                max_pixels=MAX_PIXELS,
                min_side=min_side,
            )
            image = image.resize((resized_width, resized_height), resample=Image.BICUBIC)
        
        return image
    except Exception as e:
        print(f"[Error processing image] {e}")
        return None

def qwen25vl_image_preprocess(images: Union[List[Image.Image],Image.Image], 
                              min_side:int=IMAGE_MIN_SIDE, 
                              size_factor: int = IMAGE_FACTOR,
                              force_fixed_size: Optional[Tuple[int, int]] = None, # 新增参数
                              executor: ThreadPoolExecutor=None,
                              ) -> List[Image.Image]:
    processed_images = []
    if isinstance(images, Image.Image):
        images = [images]
    if len(images) == 1:
        res = _process_one_image(images[0], min_side, size_factor, force_fixed_size)
        return [res] if res else []
    
    if executor is not None:
        futures = [executor.submit(_process_one_image, img, min_side, size_factor, force_fixed_size) 
           for img in images]
        results = [f.result() for f in futures]  # 先全部取出
        processed_images = [r for r in results if r is not None]  # 再过滤
    else:
        with ThreadPoolExecutor(max_workers=4) as temp_executor:
            futures = [temp_executor.submit(_process_one_image, img, min_side, size_factor, force_fixed_size) 
                for img in images]
            results = [f.result() for f in futures]
            processed_images = [r for r in results if r is not None]

    
    
    return processed_images


class Qwen25VLProcessor(Qwen2_5_VLProcessor):
    def __init__(self, image_processor=None, tokenizer=None, video_processor=None, chat_template=None, **kwargs):
        super().__init__(image_processor=image_processor, tokenizer=tokenizer, video_processor=video_processor, chat_template=chat_template, **kwargs)

    def __call__(self, images=None, videos=None, merge_size=2, **kwargs):
        output_kwargs = self._merge_kwargs(
            Qwen2_5_VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
    
        if images is not None:
            image_inputs = self.image_processor(images=images,  **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None

        if videos is not None:
            videos_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
    
            video_grid_thw = videos_inputs["video_grid_thw"]
        else:
            videos_inputs = {}
            video_grid_thw = None
        return BatchFeature(data={**image_inputs, **videos_inputs})


@lru_cache(maxsize=100)
def get_adaptive_pool_size(M: float, N: float, scale: float = 20) -> Tuple[int, int]:
    r = 1 / math.sqrt(scale)
    return max(1, int(np.round(M * r))), max(1, int(np.round(N * r)))


def jpeg_degrade(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


if __name__ == "__main__":
    resolution = [[360,640], [1280,720], [1366, 768],[224, 224]]

    for res in resolution:
        w, h = res[0], res[1]
        h, w = smart_resize(h, w)
        print(w,h)
    
    image = Image.open("/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/images/llava_example_cmp.png")
    new_image = qwen25vl_image_preprocess(image)
    print(new_image[0].size)
    new_image[0].save("padd.png")
