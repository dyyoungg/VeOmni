import io
import os
import re
import json
import uuid
import time
import queue
import base64
import random
import asyncio
import aiohttp
import threading
from typing import List, Optional, AsyncGenerator, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import timedelta
from functools import lru_cache

from tqdm import tqdm
from PIL import Image
import numpy as np
import av
import imageio.v3 as iio
import imageio
import openai
import soundfile as sf
from decord import VideoReader, cpu

from veomni.data.multimodal.image_utils import qwen25vl_image_preprocess
from aoss_client.client import Client as CepthClient

os.environ["TOKENIZERS_PARALLELISM"] = "false"

cepthclient = CepthClient("/mnt/afs/yangdeyu/aoss_ydy_game.conf")

BENCHMARKS = {
    # Video Benchmarks
    "mvbench": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/mvbench_test.json",
    "lvbench": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/lvbench_test.json",
    "videomme": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/videomme_test.json",
    "videommmu": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/videommmu_test.json",
    # Image Benchmarks
    "ocr": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/ocr_eval.json",
    "mmstar":"/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/VLM_Benchmark/MMStar/mmstar_test.json",
    "mmmu-dev":"/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/VLM_Benchmark/MMMU/mmmu-dev.json",
    "RealWorldQA":"/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/VLM_Benchmark/RealWorldQA/RealWorldQA_test.json",
    "CRPE": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/VLM_Benchmark/CRPE/crpe_test.json",
    "MMhalBench": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/DPO_data/MMhal_Bench/mmhal-bench_without_image.jsonl",
    "RefoMB": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/DPO_data/RefoMB/train.jsonl",
    "SyncDoc": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/img_video_instruct/OCR/SynZhOCR/syncdoc.json",
    "SyncDoc_complex": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/img_video_instruct/OCR/SynZhOCR/syncdoc_new.json",
    "chartqa": "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/VLM_Benchmark/ChartQA/chartqa_test_final.json"
}


@lru_cache(maxsize=20)
def get_video_frames(video_path, total_sample_frames, start_time=None, end_time=None, method='imageio', format="", mm_downsample_ratio=1, size_factor=28):
    start_frame = 0
    t = time.time()
    
    if os.path.exists(video_path):
        video_file = video_path
        target_input = video_path
    elif video_path.startswith("s3://"):
        video_bytes = cepthclient.Get(video_path)
        target_input = f"/dev/shm/{uuid.uuid4().hex}.mp4"
        with open(target_input, "wb") as f:
            f.write(video_bytes)
        video_file = target_input
    else:
        print('video path does not exist', video_path)
        return [], 0
    
    def get_fps(PIL_Image_object):
        """ Returns the average framerate of a PIL Image object """
        PIL_Image_object.seek(0)
        frames = 0
        while True:
            try:
                frames += 1
                PIL_Image_object.seek(PIL_Image_object.tell() + 1)
            except EOFError:
                return frames
        return None
        
    video_io = video_file
        
    is_gif = (format == "gif") or video_path.lower().endswith('.gif')
    if is_gif:
        try:
            video = iio.imread(video_io, index = None)
            frame_count = video.shape[0]
        except:
            if isinstance(video_file, bytes):
                video_io = io.BytesIO(video_file)
            else:
                video_io = video_file
            gif_obj = Image.open(video_io)
            frame_count = get_fps(gif_obj)
            video = iio.imiter(video_file)
        end_frame = frame_count
        framerate = 4
    else:
        container = av.open(video_io)
        meta_data = iio.immeta(video_file, index=None)
        if 'duration' not in meta_data:
            if container.duration is not None:
                meta_data['duration'] = container.duration / 1000000
            else:
                meta_data['duration'] = container.duration
        container.close()
        video = iio.imiter(video_file, plugin="pyav", thread_count=1)
        frame_count = int(meta_data['duration'] * meta_data['fps'])
        end_frame = frame_count
        framerate = meta_data['fps']
        if start_time != None:
            start_frame = int(meta_data['fps'] * start_time)
        if end_time != None:
            end_frame = int(meta_data['fps'] * end_time)
        frame_count = end_frame - start_frame
        if frame_count == 0:
            start_frame = 0
            end_frame = frame_count
            frame_count = end_frame - start_frame

    def get_seq_frames(total_num_frames, desired_num_frames, start_frame, end_frame, framerate):
        seg_size = float(total_num_frames - 1) / desired_num_frames
        seq = []
        for i in range(desired_num_frames):
            start = int(seg_size * i)
            end = int(seg_size * (i + 1))
            index = (start + end) // 2 + start_frame
            if index < end_frame:
                seq.append(index)
            else:
                seq.append(end_frame)
        seq = list(set(seq))
        seq.sort()
        return seq
  
    frame_seq = get_seq_frames(frame_count, min(total_sample_frames, frame_count), start_frame, end_frame, framerate)
    raw_img_list = []
    img_index = 0

    if method == 'decord' and not is_gif:
        try:
            vr = VideoReader(target_input, ctx=cpu(0))
            images_np = vr.get_batch(frame_seq).asnumpy()
            for img_np in images_np:
                raw_img_list.append(Image.fromarray(img_np).convert("RGB"))
        except ImportError:
            print("Decord not installed, falling back to imageio")
            method = 'imageio'
        except Exception as e:
            print(f"[Warning] Decord 读取失败 ({e})，自动降级为 imageio 引擎: {video_path}")
            method = 'imageio'
            raw_img_list = [] 
 
    if len(raw_img_list) == 0 or method == "imageio":
        for idx, image in enumerate(video):
            if idx == frame_seq[img_index]:
                image = Image.fromarray(image)
                raw_img_list.append(image.convert("RGB"))
                img_index += 1
                if img_index == len(frame_seq):
                    break
 
    processed_images = qwen25vl_image_preprocess(raw_img_list, mm_downsample_ratio=mm_downsample_ratio, size_factor=size_factor)
    if len(processed_images) % 2 != 0:
        processed_images.pop()

    if target_input.startswith("/dev/shm"):
        try:
            os.remove(target_input)
        except:
            pass

    return processed_images, frame_count


def get_image_frames(image_path, mm_downsample_ratio=1, size_factor=28):
    images_list = []
    if isinstance(image_path, str):
        image_paths = [image_path]

    elif isinstance(image_path, list):
        image_paths = image_path

    for path in image_paths:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                value = f.read()

        elif "s3://" in path:
            value = cepthclient.Get(image_path)
        
        if value is not None:
            img_bytes = np.frombuffer(value, np.uint8)
            buff = io.BytesIO(img_bytes)
            with Image.open(buff) as image:
                image = image.convert("RGB")
            processed_image = qwen25vl_image_preprocess(image, mm_downsample_ratio=mm_downsample_ratio, size_factor=size_factor)
            images_list.extend(processed_image)
            images_list.extend(processed_image) 

    return images_list


@dataclass
class SamplingParams:
    max_new_tokens: int = 1024
    top_k: int = 20
    top_p: float = 0.8
    temperature: float = 0.7
    repetition_penalty: float = 1.05
    
    def asdict(self) -> dict:
        return asdict(self)

def load_and_resize_image(image, target_size=(644, 364)) -> Image.Image:
    resized_image = image.resize(target_size, Image.BICUBIC)
    return resized_image

class MultiModalClient:
    def __init__(self, url: str, default_sampling_params: dict, logger=None):
        self.url = url
        self.default_sampling_params = default_sampling_params
        self._logger = logger or print

    @staticmethod
    def encode_image_to_base64(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @staticmethod
    def encode_audio_to_base64(audio: Union[np.ndarray, str, bytes]) -> str:
        if isinstance(audio, str):
            with open(audio, "rb") as f:
                audio_data = f.read()
        elif isinstance(audio, np.ndarray):
            buffer = io.BytesIO()
            np.save(buffer, audio)
            audio_data = buffer.getvalue()
        elif isinstance(audio, bytes):
            audio_data = audio
        else:
            raise ValueError(f"Unsupported audio type: {type(audio)}")
        return base64.b64encode(audio_data).decode("utf-8")

    async def generate(
        self,
        prompt: str,
        images: Optional[List[Image.Image]] = None,
        audios: Optional[List[Union[np.ndarray, str, bytes]]] = None,
        target_sizes: Optional[List[Tuple]] = None,
    ) -> AsyncGenerator[str, None]:
        
        image_data_list = []
        if images:
            for i, image in enumerate(images):
                # if target_sizes is not None:
                #     image = load_and_resize_image(image, target_sizes[i])
                # else:
                #     image = load_and_resize_image(image, target_size=(644, 364))
                image_data_list.append(self.encode_image_to_base64(image))

        audio_data_list = []
        if audios:
            for audio in audios:
                audio_data_list.append(self.encode_audio_to_base64(audio))

        payload = {
            "text": prompt,
            "sampling_params": self.default_sampling_params,
            "stream": True,
        }

        if image_data_list:
            payload["image_data"] = image_data_list if len(image_data_list) > 1 else image_data_list[0]
            
        if audio_data_list:
            payload["audio_data"] = audio_data_list if len(audio_data_list) > 1 else audio_data_list[0]
        
       
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.url, headers={"Accept": "text/event-stream"}, json=payload) as response:
                    if response.status != 200:
                        err_msg = await response.text()
                        raise Exception(f"Request failed with status {response.status}: {err_msg}")

                    prev_len = 0
                    async for line in response.content:
                        if line:
                            line_dec = line.decode('utf-8').strip()
                            if line_dec.startswith("data:"):
                                if line_dec == "data: [DONE]":
                                    break
                                json_data = json.loads(line_dec[5:].strip())
                                full_text = json_data.get("text", "")
                                new_text = full_text[prev_len:]
                                if new_text:
                                    yield new_text
                                    prev_len = len(full_text)
        except Exception as e:
            self._logger(f"Error during request: {e}")

class LLMconnector:
    def __init__(self, urls, port=18003) -> None:
        sampling_params = SamplingParams().asdict()
        print(urls)
        self.clients = [MultiModalClient(url=f"http://{url}:{port}/generate", default_sampling_params=sampling_params) for url in urls]
       
    async def generate(self, prompt, images, audios=None, sampling_params=None):
        client = random.choice(self.clients)
        try:
            full_response = ""
            # print(prompt, len(images))
            async for chunk in client.generate(prompt, images, audios):
                full_response += chunk
            return full_response
        except Exception as e:
            return f"Error: {str(e)}"

def construct_prompt(question, image_num, system_prompt=None):
    user_format = "\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
    system_format = "<|im_start|>system\n{content}<|im_end|>"
    
    final_prompt = system_format.format(content=system_prompt or "You are a helpful assistant.")
    image_prompt = "<|vision_start|>" + "<image>" * image_num + "<|vision_end|>" 
    final_prompt += image_prompt + user_format.format(content=question)
    return final_prompt

def read_json(path):
    data = []
    try:
        if path.endswith('.json'):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif path.endswith('.jsonl'):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
        else:
            raise ValueError(f"不支持的文件格式: {path}")
        print(f"Loaded {len(data)} records from {path}")
        return data
    except Exception as e:
        print(f"读取数据发生错误: {e}")
        return []


def worker_eval(args, worker_id, url, input_queue, result_queue):
    print(f"[Worker-{worker_id}] 启动，绑定 URL: {url} | Mode: {args.mode}")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        local_connector = LLMconnector(urls=[url], port=args.port)
    except Exception as e:
        print(f"[Worker-{worker_id}] 初始化 Connector 失败: {e}")
        return

    while True:
        try:
            line = input_queue.get(timeout=3) 
        except queue.Empty:
            continue
            
        if line is None:
            input_queue.task_done()
            print(f"[Worker-{worker_id}] 收到停止信号，退出。")
            break

        try:
            question = line["question"]
            system_prompt = line.get("system", line.get("system_prompt", "You are a helpful assistant."))
            
            if args.mode == "video":
                media_path = line.get("path", line.get("video_path", ""))
                images_list, _ = get_video_frames(
                    media_path,
                    args.eval_frames,
                    line.get("start", None),
                    line.get("end", None),
                    'decord',
                    "",
                    args.mm_downsample_ratio,
                    args.size_factor
                )
            else:
                media_path = line.get("image_path", line.get("image", ""))
                images_list = get_image_frames(media_path, args.mm_downsample_ratio, args.size_factor)
           
            prompt = construct_prompt(question, image_num=len(images_list), system_prompt=system_prompt)
           
            sample_param = SamplingParams(temperature=0.7).asdict()
            answer = loop.run_until_complete(
                local_connector.generate(prompt, images=images_list, sampling_params=sample_param)
            )
            
            if args.dataset_name not in ["MMhalBench", "RefoMB"]:
                match = re.search(r'([A-Z])', answer)
                pred_option = match.group(1) if match else ""
            else:
                pred_option = answer.replace("The answer is ", "").replace(".", "")
            
            result = {
                "id": line.get("id", ""),
                "gt": line.get("answer_option", ""),
                "pred": pred_option,
                "raw_answer": answer,
                "question": question,
                "media_path": media_path,
                "category": line.get("category", line.get("task", "unknown")),
                "status": "success"
            }
        except Exception as e:
            media_path = line.get("path", line.get("image_path", "unknown"))
            print(f"[Worker-{worker_id}] 处理失败 {media_path}: {e}")
            result = {
                "status": "error", 
                "error_msg": str(e), 
                "media_path": media_path, 
                "category": line.get("category", line.get("task", "unknown"))
            }

        result_queue.put(result)
        input_queue.task_done()

def writer_logic(result_queue, output_path, total_tasks, pbar):
    print(f"[Writer] 启动，结果将实时写入: {output_path}")
    correct_count = 0
    processed_count = 0
    category_stats = {}
    
    with open(output_path, "w", encoding="utf-8", buffering=1) as f:
        while True:
            item = result_queue.get()
            if item is None:
                result_queue.task_done()
                break
            
            if item.get("status") == "success":
                processed_count += 1
                cat = item.get("category", "unknown")
                if cat not in category_stats:
                    category_stats[cat] = {"total": 0, "correct": 0}
                
                category_stats[cat]["total"] += 1
                if item["gt"] == item["pred"]:
                    correct_count += 1
                    category_stats[cat]["correct"] += 1
                    item["is_correct"] = True
                else:
                    item["is_correct"] = False
                    
            try:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            except Exception as e:
                pass
            
            with pbar.get_lock():
                pbar.update(1)
                if processed_count > 0:
                    pbar.set_postfix({"Acc": f"{correct_count/processed_count:.2%}"})
            
            result_queue.task_done()

    print("\n--- 最终评测结果统计 ---")
    overall_acc = correct_count / processed_count if processed_count > 0 else 0
    print(f"总体准确率: {overall_acc:.4f} ({correct_count}/{processed_count})")
    print("各类别准确率:")
    for cat, stats in category_stats.items():
        cat_acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {cat}: {cat_acc:.4f} ({stats['correct']}/{stats['total']})")


def main(args, urls_list):
    # dataset_path 优先；否则 dataset_name 查预定义表；都没给则报错
    if args.dataset_path:
        if not os.path.exists(args.dataset_path):
            raise FileNotFoundError(f"Custom dataset path not found: {args.dataset_path}")
        target_path = args.dataset_path
        if not args.dataset_name:
            args.dataset_name = os.path.splitext(os.path.basename(target_path))[0]
    elif args.dataset_name:
        key = args.dataset_name.lower() if args.dataset_name.lower() in BENCHMARKS else args.dataset_name
        if key not in BENCHMARKS:
            raise ValueError(
                f"'{args.dataset_name}' is not in predefined BENCHMARKS. "
                f"Available: {list(BENCHMARKS.keys())}. Or use --dataset_path for a custom file."
            )
        target_path = BENCHMARKS[key]
    else:
        raise ValueError("Must specify either --dataset_name or --dataset_path.")

    raw_data = read_json(target_path)
    
    if args.mode == "video" and "category" in raw_data[0] and hasattr(args, 'dataset_name') and args.dataset_name:
        raw_data = [line for line in raw_data if line.get("category", "").lower() == args.dataset_name.lower()]

    task_list = []
    for line in raw_data:
        question = line.get("question", "")
        answer = line.get("answer", line.get("gt_answer", ""))
        
        if args.dataset_name not in ["MMhalBench", "RefoMB"]:
            answer_idx = ""
            if "candidates" in line:
                for idx, c in enumerate(line['candidates']):
                    option_char = chr(ord('A') + idx)
                    question += f"\n({option_char}) {c}"
                    if c == answer:
                        answer_idx = option_char
            
       
            question += "\nPlease answer the question use a single option like (A) or (B) or...etc." if args.mode == "video" else "\nAnswer the question using single option word (e.g. A or B or C or D) and nothing else."
            line["answer_option"] = answer_idx
        else:
            line["answer_option"] = str(answer)
            
        line["question"] = question
        task_list.append(line)

    random.shuffle(task_list)
    total_tasks = len(task_list)
    print(f"Total tasks prepared for {args.mode} evaluation: {total_tasks}")

    task_queue = queue.Queue()
    result_queue = queue.Queue()
    for task in task_list:
        task_queue.put(task)

    # 结果路径
    if hasattr(args, 'save_jsonl_path') and args.save_jsonl_path:
        output_file = args.save_jsonl_path
    else:
        output_file = os.path.join(args.save_dir, f"{args.model_type}_{args.dataset_name}_results.jsonl")
        
    os.makedirs(os.path.dirname(output_file) or args.save_dir, exist_ok=True)
    
    pbar = tqdm(total=total_tasks, desc=f"Evaluating {args.mode.capitalize()}s")
    writer = threading.Thread(target=writer_logic, args=(result_queue, output_file, total_tasks, pbar))
    writer.daemon = True
    writer.start()

    workers = []
    worker_id = 0
    
    # 针对每个传入的 URL 启动指定数量的 Worker 线程
    for url in urls_list:
        for _ in range(args.workers_per_url):
            t = threading.Thread(target=worker_eval, args=(args, worker_id, url, task_queue, result_queue))
            t.daemon = True
            t.start()
            workers.append(t)
            worker_id += 1

    # 在所有实际任务排队完成后，放入与 Worker 数量对等的结束信号
    for _ in workers:
        task_queue.put(None)

    # 等待所有任务处理完毕
    task_queue.join()
    
    for t in workers:
        t.join()
        
    result_queue.put(None)
    result_queue.join() 
    writer.join()
    pbar.close()
    print("Eval Finished.")

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["video", "image"], required=True, help="评测模式：视频或图片")
    
    parser.add_argument("--urls", type=str, nargs="+", default=["127.0.0.1"], help="后端服务的 IP 或域名列表，空格分隔")
    parser.add_argument("--workers_per_url", type=int, default=1, help="每个 URL 绑定的并发线程数 (并发度)")
    
    # 路径与数据相关（二选一：dataset_name 走预定义路径，dataset_path 走自定义 JSON）
    parser.add_argument("--dataset_name", type=str, default="", help="预定义数据集名称，可选: " + ", ".join(BENCHMARKS.keys()))
    parser.add_argument("--dataset_path", type=str, default="", help="自定义 JSON/JSONL 文件路径，优先级高于 dataset_name")
    
    parser.add_argument("--save_dir", type=str, default="./data")
    parser.add_argument("--save_jsonl_path", type=str, default="")
    parser.add_argument("--eval_frames", type=int, default=64, help="视频抽帧数量")
    parser.add_argument("--model_type", type=str, default="llava")
    parser.add_argument("--port", type=int, default=18003)
    parser.add_argument("--mm_downsample_ratio", type=int, default=1, help="mm_downsample_ratio for image preprocessing")
    parser.add_argument("--size_factor", type=int, default=28, help="IMAGE_FACTOR for image preprocessing")
    args = parser.parse_args()
    
    main(args, args.urls)