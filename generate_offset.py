import os
import json
import numpy as np
import glob
from tqdm import tqdm
import random

def get_jsonl_offsets(file_path):
 
    offsets = []
    curr_offset = 0
    with open(file_path, 'rb') as f:
        offsets.append(np.uint64(0)) 
        while True:
            line = f.readline()
            if not line:
                break
            curr_offset += len(line)
            offsets.append(np.uint64(curr_offset))
    
    if len(offsets) > 0:
        offsets.pop()
    return offsets


def generate_index_files(file_paths: str, output_dir: str):

    all_sampled_records = []
    sample_ratio = 1.0
    file_mapping = []
    for f_idx, path in enumerate(file_paths):
        print(f"Scanning: {path}")
        file_mapping.append(path)
        offsets = get_jsonl_offsets(path)
        
        num_samples = int(len(offsets) * sample_ratio)
        sampled_indices = random.sample(range(len(offsets)), num_samples)
        
        for idx in sampled_indices:
            all_sampled_records.append([f_idx, offsets[idx]])
                    
    mapping_path = os.path.join(output_dir, "file_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(file_mapping, f, ensure_ascii=False, indent=2)
        
    
    offsets_path = os.path.join(output_dir, "offsets.npy")
    
    offsets_array = np.array(offsets, dtype=np.uint64)
    np.save(offsets_path, all_sampled_records)
    
    print(f"Index generated! Total files: {len(file_mapping)}, Total samples: {len(offsets_array)}")


def load_sample(index_dir: str, global_index: int) -> dict:
  
    mapping_path = os.path.join(index_dir, "file_mapping.json")
    offsets_path = os.path.join(index_dir, "offsets.npy")

    with open(mapping_path, "r", encoding="utf-8") as f:
        file_mapping = json.load(f)
    full_offsets = np.load(offsets_path, mmap_mode='r')
    
    if global_index < 0 or global_index >= len(full_offsets):
        raise IndexError(f"Index {global_index} 越界！总数据量为: {len(full_offsets)}")
        

    file_id, offset = full_offsets[global_index]
    file_path = file_mapping[int(file_id)]
    

    with open(file_path, "r", encoding="utf-8") as f:
        f.seek(int(offset))
        line = f.readline()
        
    return json.loads(line)


if __name__ == "__main__":
    INDEX_DIR = "/mnt/afs/yangdeyu/GameMLLM/VeOmni-Dev"  # 替换成你的目录

    file_paths = ["/mnt/afs/yangdeyu/GameMLLM/llava_dev/LLaVA_hub/data/stage2/train_0.jsonl",
                 
                  ]
    generate_index_files(file_paths=file_paths, output_dir=INDEX_DIR)
    TEST_INDEX = 99                  
    
    try:
        sample = load_sample(INDEX_DIR, TEST_INDEX)
        print("✅ 读取成功！数据预览：")
        # 仅打印前 200 个字符防刷屏
        print(str(sample)) 
    except Exception as e:
        print(f"❌ 读取失败: {e}")
