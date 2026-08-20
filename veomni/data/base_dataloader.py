import gc
import io
import json
import math
import os
import queue
import random
import re
import signal
import socket
import threading
from abc import abstractmethod
from typing import Dict, List, Optional, Sequence, Union
import time

import numpy as np
import requests
import torch
from PIL import Image
from flask import Flask, request
from pydub import AudioSegment
from torch.utils.data import Dataset
import transformers
import soundfile as sf
import resampy
import librosa
import imageio.v3 as iio
from imageio.core import Request


from veomni.utils.helper import read_data
from veomni.utils.logging import get_logger
from veomni.utils.constants import _CHAT_TEMPLATES, REMOTE_SERVER_PORT


TIMEOUT = 30
logger = get_logger(__name__)

class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException()


class BaseDataLoader:
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args,
        training_args,
        model_args,
        eval_mode: bool,
    ):
        self.eval_mode = eval_mode
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        self.data_args = data_args
        self.training_args = training_args
        self.model_args = model_args
        self.tokenizer = tokenizer
        self.tokenizer.add_bos_token = False

        self.dataloader_num_workers = training_args.dataloader_num_workers
        self.debug = False
        self.is_launched = False
        self.remote_data_index = torch.multiprocessing.Value("i", 0)

        self.data_list: List = []
        self.file_mapping = None
        data_path = data_args.eval_path if eval_mode else data_args.train_path
        self.load_data(data_path)
        

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self, data_path: str) -> None:
        if self.eval_mode:
            return
        if getattr(self.training_args, "use_fake_data", False):
            num_samples = getattr(self.training_args, "fake_data_num_samples", 10000)
            self.data_list = list(range(num_samples))
            logger.info(f"[FakeData] rank {self.rank}: use_fake_data=True, generated {num_samples} fake sample indices")
            return
        if self.training_args.remote_dataloader:
            assert os.path.exists(getattr(self.data_args, "offset_file_path", "")), "remote dataloader need offset file. Please check the offset file path."
            assert os.path.exists(getattr(self.data_args, "file_maping_path", "")), "remote dataloader need filemaping file. Please check the file_maping file path."
           
            mapping_path = getattr(self.data_args, "file_maping_path", "")
            offsets_path = getattr(self.data_args, "offset_file_path", "")
            
            if mapping_path and mapping_path.endswith('.json'):
                try:
                    with open(mapping_path, 'r', encoding='utf-8') as f:
                        self.file_mapping = json.load(f)
                except Exception as e:
                    logger.error(f"Failed to load offset_file as JSON mapping: {e}")
                    self.file_mapping = {}
            else:
                self.file_mapping = mapping_path

            if offsets_path.endswith('.npy'):
                self.data_list = np.load(offsets_path, mmap_mode='r')
            elif offsets_path.endswith('.json'):
                with open(offsets_path, 'r') as f:
                    self.data_list = json.load(f)
        else:

            if not self.data_args.offline_dataset_split:
                random.seed(233)
                num_epochs = int(self.training_args.num_train_epochs)
                chunk_size = None  # computed after read

                raw = read_data(data_path=data_path)
                chunk_size = len(raw) // self.world_size
                self.data_list = []
                for _ in range(num_epochs):
                    random.shuffle(raw)
                    self.data_list.extend(
                        raw[self.rank * chunk_size: (self.rank + 1) * chunk_size]
                    )
                

                split_label = "test" if self.eval_mode else "train"
                if split_label == "train":
                    print(f"{self.rank}: {split_label} data size {len(self.data_list)}")
            else:
                path = os.path.join(data_path, f"train_{self.rank}.jsonl")
                try:
                    self.data_list.extend(read_data(data_path=path))
                except Exception as e:
                    print(f"read jsonl data error: {e}")
                print(f"{self.rank}: training data size {len(self.data_list)}")

            self.data_list = [json.dumps(d).encode() for d in self.data_list]


    def __len__(self) -> int:
        return len(self.data_list)


    def launch(self) -> None:
        """Set up worker processes and I/O queues.  Designed to support
        resume-from-checkpoint in the trainer."""
        if self.is_launched:
            return
        self.end_signal = False
        self.is_launched = False

        # inter-process queues
        self.data_queue = torch.multiprocessing.Queue()
        self.result_queue = torch.multiprocessing.Queue()
        self.batch_data_queue = queue.Queue(maxsize=1)

        # fetch thread (GPU-side collation)
        self.pin_memory_thread_done_event = threading.Event()
        self.fetch_data_thread = threading.Thread(
            target=self.fetch_data_loop, daemon=True
        )
        self.fetch_data_thread.start()

        # worker processes / threads
        self.worker_processes: List = []
        self.worker_status_event: List = []
        self.workers_done_event = torch.multiprocessing.Event()

        # batch accumulation state
        self.new_caption_len = 0
        self.new_input_ids: List = []
        self.new_labels: List = []
        self.new_images_list: List = []
        self.new_images_thw: List = []
        self.new_data_lens: List = []
        self.audio_temp_list: List = []
        self.audio_max_duration = self.model_args.audio_max_duration

        self.start_worker()

        # remote data-server (rank-0 only)
        self.remote_server_process = None
        if getattr(self.training_args, "use_fake_data", False):
            # fake data mode: worker_loop will generate data internally
            pass
        elif self.training_args.remote_dataloader:
            if self.rank == 0:
                self.remote_server_process = torch.multiprocessing.Process(
                    target=self.remote_server_loop
                )
                self.remote_server_process.start()
        else:
            for d in self.data_list:
                self.data_queue.put(d)
        time.sleep(2.0)
        self.is_launched = True

    def start_worker(self) -> None:
        if self.dataloader_num_workers == 0:
            status_event = torch.multiprocessing.Event()
            p = threading.Thread(
                target=self.worker_loop, args=(status_event,), daemon=True
            )
            p.start()
            self.worker_status_event.append(status_event)
            self.worker_processes.append(p)
        else:
            for _ in range(self.dataloader_num_workers):
                status_event = torch.multiprocessing.Event()
                p = torch.multiprocessing.Process(
                    target=self.worker_loop, args=(status_event,), daemon=True
                )
                p.start()
                self.worker_status_event.append(status_event)
                self.worker_processes.append(p)

    def remote_server_loop(self) -> None:
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)  # 只打印 Error 级别的日志，屏蔽 200 OK
        log.disabled = True
        app = Flask(__name__)

        @app.route("/ask_data", methods=["POST"])
        def ask_data():
            ret = {"index": self.remote_data_index.value}
            self.remote_data_index.value += 1
            return ret
        master_addr = os.environ.get("MASTER_ADDR")
        if master_addr is None:
            # single node
            master_addr = socket.gethostname()
        app.run(
            host=master_addr,
            port=REMOTE_SERVER_PORT,
            debug=False,
            use_reloader=False,
        )

    def _drain_queue(self, q, use_nowait: bool = True) -> None:
        """Empty a queue, swallowing all exceptions."""
        try:
            while True:
                if use_nowait:
                    q.get_nowait()
                else:
                    if q.qsize():
                        q.get()
                    else:
                        break
        except Exception:
            pass

    def close(self) -> None:
        if not self.is_launched:
            return

        self.end_signal = True
        self.workers_done_event.set()
        if self.result_queue is not None:
            self.result_queue.cancel_join_thread()
        if self.data_queue is not None:
            self.data_queue.cancel_join_thread()

        self._drain_queue(self.batch_data_queue, use_nowait=False)
        if self.result_queue is not None:
            self._drain_queue(self.result_queue)
        if self.data_queue is not None:
            self._drain_queue(self.data_queue)
        
        if self.fetch_data_thread is not None:
            self.fetch_data_thread.join(timeout=2)
            self.fetch_data_thread = None

        for p in self.worker_processes:
            try:
                p.join(timeout=5)
                if p.is_alive():
                    logger.warning(f"rank {self.rank} Worker {p.pid} did not exit, terminating...")
                    p.terminate()
                    p.join()
            except Exception as e:
                logger.warning(f"[close] join worker failed: {e}")
        self.worker_processes = []

        if self.result_queue is not None:
            self.result_queue.close()
            self.result_queue = None

        if self.data_queue is not None:
            self.data_queue.close()
            self.data_queue = None

        if self.rank == 0 and self.remote_server_process is not None:
            try:
                self.remote_server_process.terminate()
                self.remote_server_process.join()
            except Exception as e:
                print(f"[close] remote_server_process stop failed: {e}")
            self.remote_server_process = None
        logger.info(f"rank {self.rank} closed dataloader successful!")
        self.is_launched = False

    def check_all_workers_done(self) -> bool:
        if not self.is_launched:
            return False
        return all(e.is_set() for e in self.worker_status_event)

    def __iter__(self):
        while True:
            try:
                data = self.batch_data_queue.get(timeout=1)
                # print("data data data data data data", data)
                yield data
            except queue.Empty:
                if self.check_all_workers_done():
                    time.sleep(5.0)
                    if self.batch_data_queue.empty():
                        return 
                    else:
                        continue 

    @abstractmethod
    def worker_loop(self, status_event):
        raise NotImplementedError

    @abstractmethod
    def fetch_data_loop(self):
        raise NotImplementedError

    def process_subtitle_srt(self, subtitle_bytes: bytes) -> List[Dict]:
        def parse_timestamp(ts: str) -> float:
            h, m, rest = ts.split(":")
            s, ms = rest.split(",")
            return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

        subtitles = []
        for block in subtitle_bytes.decode("utf-8").split("\n\n"):
            if not block.strip():
                continue
            lines = block.split("\n")
            start, end = lines[1].split(" --> ")
            subtitles.append(
                {
                    "index": lines[0],
                    "start_time": parse_timestamp(start),
                    "end_time": parse_timestamp(end),
                    "text": " ".join(lines[2:]),
                }
            )
        return subtitles

    def process_subtitle_json(self, json_file: Dict) -> List[Dict]:
        subtitle_list = []
        for i, s in enumerate(json_file["srt_dict"]):
            text = s[0].strip()
            if text.startswith("[") and text.endswith("]"):
                continue
            if self.data_args.filter_subtitle:
                text = re.sub(r"\[.*?\]", "", text)
            start_time = s[1]
            end_time = s[1] + s[2] if len(s) == 3 else start_time
            if text:
                subtitle_list.append(
                    {
                        "index": i,
                        "start_time": start_time,
                        "end_time": end_time,
                        "text": text,
                    }
                )
        return subtitle_list

    # ------------------------------------------------------------------
    # Image preprocessing (legacy CLIP path, kept for back-compat)
    # ------------------------------------------------------------------

    def preprocess_image(self, image: Image.Image) -> List[torch.Tensor]:
        raise NotImplementedError


    def build_inputs_token(
        self,
        input_str: str = "",
        input_type: Optional[str] = None,
        return_tensor: bool = True,
    ):
        arc = self.model_args.model_arc
        templates = _CHAT_TEMPLATES.get(arc)
        if templates is None:
            raise NotImplementedError(f"Unsupported model_arc: {arc!r}")

        template = templates.get(input_type)
        if template is None:
            raise ValueError(
                f"Unknown input_type {input_type!r}. "
                f"Valid options: {list(templates.keys())}"
            )

        if input_type == "assistant_prefix":
            text = template
        else:
            text = template.format(input_str)

        tokens = self.tokenizer(text)["input_ids"]
        return torch.tensor(tokens, dtype=torch.long) if return_tensor else tokens


    def safe_call_with_timeout(self, func, timeout: int, *args, **kwargs):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        try:
            return func(*args, **kwargs)
        except TimeoutException:
            self.logger.error(
                f"[safe_call_with_timeout] Timeout after {timeout}s: {func.__name__}"
            )
            return None
        except Exception as e:
            self.logger.error(
                f"[safe_call_with_timeout] Exception in {func.__name__}: {e}"
            )
            return None
        finally:
            signal.alarm(0)

    def get_audio_data_list(self, data_dict: Dict) -> List[torch.Tensor]:
        """Load all audio tensors referenced in *data_dict*."""
        audio_field = data_dict["audio"]
        paths = [audio_field] if isinstance(audio_field, str) else audio_field

        result = []
        for path in paths:
            tensor, _ = self.get_audio_from_local_or_remote(path)
            if tensor is not None:
                result.append(tensor)
        return result

  