import os
import json
import numpy as np
import glob
from tqdm import tqdm
import random
import types
from typing import List, Union, Tuple, Dict,Generator, Any, Iterable



def stream_read_json_data(file_path: str) -> Generator[Any, None, None]:
    """
    流式读取 json 或 jsonl 文件，按条目 yield 数据。
    适合处理 GB 级别的大文件，防止内存溢出。
    """
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
        # 标准 JSON 只能全量加载后再逐个 yield（如果顶层是列表）
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
    
    参数:
        data: 要写入的数据（支持 List, Dict 或 Generator）。
        file_path: 输出文件路径（必须以 .json 或 .jsonl 结尾）。
        mode: 写入模式（仅对 jsonl 有效，'w' 覆盖，'a' 追加）。
        indent: 缩进量（仅对 json 有效）。
    """
    # 自动创建父级目录
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    if file_path.endswith('.jsonl'):
        count = 0
        with open(file_path, mode, encoding='utf-8') as f:
            for record in data:
                # 紧凑输出，确保单行
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                count += 1
        print(f"✅ 成功流式写入 {count} 条数据至: {file_path} (JSONL)")
        
    elif file_path.endswith('.json'):
        # 强制拦截：如果传入的是生成器，必须先物化为列表才能供 json.dump 使用
        if isinstance(data, (types.GeneratorType, map, filter)):
            print("⚠️ 警告: 检测到生成器输入且目标格式为 .json。正在将其转换为全量列表，如果数据量极大可能导致 OOM。")
            data = list(data)
            
        # 标准 JSON 通常不支持 'a' 追加模式，强转为 'w'
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
            
        # 简单统计长度用于打印（如果是列表或字典）
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
        # 假设 json 文件是一个大 list: [{}, {}, ...]
        data = json.load(f_in)
        
    with open(jsonl_path, 'w', encoding='utf-8') as f_out:
        for record in data:
            # 必须 ensure_ascii=False 保证中文正常存储，同时紧凑输出
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
            
    print(f"Conversion done! Saved to {jsonl_path}")
    return jsonl_path

def get_dataset_statistics(file_configs: List[Tuple[str, float]], save_path: str = None) -> Dict:
   
    stats_list = []
    total_original_lines = 0
    total_sampled_lines = 0
    total_size_bytes = 0
    
    # 打印表头
    print(f"{'File Name':<55} | {'Size(MB)':<10} | {'Original':<10} | {'Ratio':<6} | {'Sampled':<10}")
    print("-" * 105)
    
    for path, ratio in file_configs:
        if not os.path.exists(path):
            print(f"Warning: File not found - {path}")
            continue
            
        file_size = os.path.getsize(path)
        total_size_bytes += file_size
        
        # 高效统计行数（比 f.readlines() 或 json.load() 快得多）
        line_count = 0
        with open(path, 'rb') as f:
            for _ in f:
                line_count += 1
                
        expected_sampled_count = int(line_count * ratio)
        
        total_original_lines += line_count
        total_sampled_lines += expected_sampled_count
        
        size_mb = round(file_size / (1024 * 1024), 2)
        file_name = os.path.basename(path)
        
        # 截断过长的文件名以保证终端对齐
        display_name = file_name if len(file_name) <= 53 else file_name[:50] + "..."
        
        # 打印单行统计信息
        print(f"{display_name:<55} | {size_mb:<10.2f} | {line_count:<10} | {ratio:<6} | {expected_sampled_count:<10}")
        
        stats_list.append({
            "file_path": path,
            "file_name": file_name,
            "file_size_MB": size_mb,
            "original_lines": line_count,
            "sample_ratio": ratio,
            "expected_sampled_lines": expected_sampled_count
        })
        
    summary = {
        "total_files": len(stats_list),
        "total_size_GB": round(total_size_bytes / (1024 ** 3), 2),
        "total_original_lines": total_original_lines,
        "total_expected_sampled_lines": total_sampled_lines,
        "details": stats_list
    }
    
    print("-" * 105)
    print(f"Total Size: {summary['total_size_GB']} GB | Original Lines: {total_original_lines} | Sampled Lines: {total_sampled_lines}")
    
    # 选择性落盘统计结果
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nStatistics saved to {save_path}")
        
    return summary



def get_jsonl_offsets(file_path):
    # 定义缓存文件路径
    cache_path = f"{file_path}_offset.json"
    
    if os.path.exists(cache_path):
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
):
    all_sampled_records = []
    file_mapping = []
    
  
    for f_idx, (path, ratio) in enumerate(file_paths):
        print(f"Scanning: {path} | Sample Ratio: {ratio}")

        if path.endswith('.json'):
            path = convert_json_to_jsonl(path)

        file_mapping.append(path)
        
        offsets = get_jsonl_offsets(path)
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
                    
    mapping_path = os.path.join(output_dir, "file_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(file_mapping, f, ensure_ascii=False, indent=2)
        
    offsets_path = os.path.join(output_dir, "offsets.npy")
    
    offsets_array = np.array(all_sampled_records, dtype=np.uint64)
    np.save(offsets_path, offsets_array)
    
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




def generate_game_text_train():
    INDEX_DIR = "/mnt/afs/yangdeyu/GameMLLM/VeOmni-Dev/exp_data/0511_stage1_puretext"  # 替换成你的目录

    os.makedirs(INDEX_DIR, exist_ok=True)

    file_paths =[
        ['/mnt/afs/zhouhang/data/wiki_data/baidubaike/4_16_splitgather_oneAI.jsonl', 1.0],
        ['/mnt/afs/zhouhang/data/wiki_data/baidubaike/4_28_500w.jsonl', 1.0],
        ['/mnt/afs/zhouhang/data/wiki_data/bili_wiki/0410_bili_wiki--0410gather_oneAI.jsonl', 1.0],
        ['/mnt/afs/zhouhang/data/wiki_data/huiji_wiki/0410_huiji_wiki--0410gather_oneAI.jsonl', 1.0],
        ['/mnt/afs/zhouhang/data/wiki_data/video_md/train_data/4_29.jsonl', 1.0],
        ['/mnt/afs/zhouhang/data/wiki_data/youmin_all/train_data/0429--0429_qagather_oneAI.jsonl', 1.0],
        ['/mnt/afs/zhouhang/data/wiki_data/youmin_youxia/0410_new_filter_merge--0410gather_oneAI.jsonl', 1.0],
        ['/mnt/afs/zhouhang/data/wiki_data/zhihu/output/0424.jsonl', 1.],
        ['/mnt/afs/zhouhang/data/wiki_data/zhihu/3_9_zhihu_23w_gather.jsonl', 1.],
        ['/mnt/afs/zhouhang/data/wiki_data/igdb/hand_data/3_6_hand_igdb.jsonl', 1.],
        ['/mnt/afs/zhouhang/data/wiki_data/igdb/3_6_igdb_3w_gather_fix.jsonl', 1.],
        ['/mnt/afs/zhouhang/data/wiki_data/wiki_zh/3_9_hand_data/3_9_150w_wiki_markdown.jsonl', 1.0],
        ['/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/common_instruct/0724_en_all_329w.jsonl', 1.0],
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/common_instruct/0724_zh_all_494w.jsonl", 0.8],
        ['/mnt/afs/jiayi/data/mm_data/game_new/train_data/20260217_lol_add_text_check.jsonl', 1.0]
        ]
    
    file_paths = [(path, ratio) for path, ratio in file_paths]

    # get_dataset_statistics(file_paths)
    generate_index_files(file_paths=file_paths, output_dir=INDEX_DIR)
    TEST_INDEX = 99                  
    
    try:
        sample = load_sample(INDEX_DIR, TEST_INDEX)
        print("✅ 读取成功！数据预览：")
        # 仅打印前 200 个字符防刷屏
        print(str(sample)) 
    except Exception as e:
        print(f"❌ 读取失败: {e}")





if __name__ == "__main__":
    generate_game_text_train()

   

    

