import os
import json
import numpy as np
import glob
from tqdm import tqdm
import random
import types
from typing import List, Union, Tuple, Dict,Generator, Any, Iterable



def stream_read_json_data(file_path: str) -> Generator[Any, None, None]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件未找到: {file_path}")

    if file_path.endswith('.jsonl'):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"解析错误: 文件 {file_path} 第 {line_num} 行 - {e}")
                    
    elif file_path.endswith('.json'):
      
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        yield item
                else:
                    yield data  # 如果顶层是字典，直接 yield 整个字典
            except json.JSONDecodeError as e:
                print(f"解析错误: 文件 {file_path} - {e}")
    else:
        raise ValueError(f"不支持的文件扩展名，仅支持 .json 或 .jsonl: {file_path}")

def save_json_data(
    data: Union[Iterable[Any], Dict], 
    file_path: str, 
    mode: str = 'w', 
    indent: int = 2
) -> str:
    """
    统一写入 JSON 和 JSONL 的函数，根据文件扩展名自动选择策略。
    """
   
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    if file_path.endswith('.jsonl'):
        count = 0
        with open(file_path, mode, encoding='utf-8') as f:
            for record in data:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                count += 1
        print(f"✅ 成功流式写入 {count} 条数据至: {file_path} (JSONL)")
        
    elif file_path.endswith('.json'):
    
        if isinstance(data, (types.GeneratorType, map, filter)):
            print("⚠️ 警告: 检测到生成器输入且目标格式为 .json。正在将其转换为全量列表，如果数据量极大可能导致 OOM。")
            data = list(data)
            
      
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
            
        length = len(data) if isinstance(data, (list, dict)) else 1
        print(f"✅ 成功全量写入数据 (长度/Keys: {length}) 至: {file_path} (JSON)")
        
    else:
        raise ValueError(f"❌ 不支持的文件扩展名: {file_path}。仅支持 .json 或 .jsonl")
        
    return file_path


def convert_json_to_jsonl(json_path: str, jsonl_path: str = None):
 
    if jsonl_path is None:
        jsonl_path = json_path.rsplit('.', 1)[0] + '.jsonl'
        
    print(f"Converting {json_path} to {jsonl_path}...")
    
    with open(json_path, 'r', encoding='utf-8') as f_in:
        data = json.load(f_in)
        
    with open(jsonl_path, 'w', encoding='utf-8') as f_out:
        for record in data:
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
            
    print(f"Conversion done! Saved to {jsonl_path}")
    return jsonl_path

def get_jsonl_offsets(file_path, regen_offset=False):
    
    cache_path = f"{file_path}_offset.json"
    
    if os.path.exists(cache_path) and not regen_offset:
        print(f"Loading cached offsets from: {cache_path}")
        with open(cache_path, 'r', encoding='utf-8') as f:
            cached_offsets = json.load(f)
       
        return [np.uint64(x) for x in cached_offsets]
 
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
        
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump([int(x) for x in offsets], f)
        
    return offsets


def generate_index_files(
    file_paths: List[str], 
    output_dir: str, 
    regen_offset:bool=False,
    split_ratios: List[float] = [1.0], # 传入比例列表
    split_names: List[str] = ["train"] # 对应的输出文件后缀
):
    all_sampled_records = []
    file_mapping = []
    
  
    for f_idx, (path, ratio) in enumerate(file_paths):
        print(f"Scanning: {path} | Sample Ratio: {ratio}")

        if path.endswith('.json'):
            path = convert_json_to_jsonl(path)

        file_mapping.append(path)
        
        offsets = get_jsonl_offsets(path, regen_offset=regen_offset)
        total_lines = len(offsets)
        
        if total_lines == 0 or ratio <= 0:
            continue
            
        num_samples = int(total_lines * ratio)
        
      
        if ratio <= 1.0:  
            sampled_indices = random.sample(range(total_lines), num_samples)
        else:
         
            full_copies = num_samples // total_lines
            remainder = num_samples % total_lines
            
            sampled_indices = list(range(total_lines)) * full_copies
            sampled_indices.extend(random.sample(range(total_lines), remainder))
            
            random.shuffle(sampled_indices)
        
        for idx in sampled_indices:
            all_sampled_records.append([f_idx, offsets[idx]])
    
    random.shuffle(all_sampled_records)
    mapping_path = os.path.join(output_dir, "file_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(file_mapping, f, ensure_ascii=False, indent=2)

    total_records = len(all_sampled_records)
    
    
    ratio_sum = sum(split_ratios)
    normalized_ratios = [r / ratio_sum for r in split_ratios]
    
    split_sizes = [int(total_records * r) for r in normalized_ratios]
    
   
    split_sizes[0] += total_records - sum(split_sizes)
    
    print(f"Index generated! Total files: {len(file_mapping)}, Total samples: {total_records}")
    

    start_idx = 0
    for i, size in enumerate(split_sizes):
        if size == 0:
            continue
            
        end_idx = start_idx + size
        split_data = all_sampled_records[start_idx:end_idx]
        
       
        name_suffix = split_names[i] if i < len(split_names) else f"split_{i}"
        offsets_path = os.path.join(output_dir, f"offsets_{name_suffix}.npy")
        
       
        offsets_array = np.array(split_data, dtype=np.uint64)
        np.save(offsets_path, offsets_array)
        print(f"  -> Saved {name_suffix}: {size} samples to {offsets_path}")
        start_idx = end_idx    
    
def load_sample(index_dir: str, global_index: int, full_offsets:list=None, file_mapping:list=None, split_name="train") -> dict:
  
    mapping_path = os.path.join(index_dir, "file_mapping.json")
    offsets_path = os.path.join(index_dir, f"offsets_{split_name}.npy")

    if file_mapping is None:
        with open(mapping_path, "r", encoding="utf-8") as f:
            file_mapping = json.load(f)

    if full_offsets is None:
        full_offsets = np.load(offsets_path, mmap_mode='r')
    
    if global_index < 0 or global_index >= len(full_offsets):
        raise IndexError(f"Index {global_index} 越界！总数据量为: {len(full_offsets)}")
        

    file_id, offset = full_offsets[global_index]
    file_path = file_mapping[int(file_id)]
    

    with open(file_path, "r", encoding="utf-8") as f:
        f.seek(int(offset))
        line = f.readline()
        
    return json.loads(line)




def generate_game_text_train():
    INDEX_DIR = ""  # 替换成你的目录

    os.makedirs(INDEX_DIR, exist_ok=True)

    file_paths =[
        ('/mnt/afs/zhouhang/data/wiki_data/baidubaike/4_16_splitgather_oneAI.jsonl', 1.0),


        ]

    generate_index_files(file_paths=file_paths, output_dir=INDEX_DIR, regen_offset=False)
    TEST_INDEX = 99                  
    
    try:
        sample = load_sample(INDEX_DIR, TEST_INDEX)
        print("✅ 读取成功！数据预览：")
        print(str(sample)) 
    except Exception as e:
        print(f"❌ 读取失败: {e}")


if __name__ == "__main__":
    generate_game_text_train()