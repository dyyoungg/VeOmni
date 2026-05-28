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



def get_jsonl_offsets(file_path, regen_offset=False):
    # 定义缓存文件路径
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
    regen_offset:bool=False
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
        
    offsets_path = os.path.join(output_dir, "offsets.npy")
    
    offsets_array = np.array(all_sampled_records, dtype=np.uint64)
    np.save(offsets_path, offsets_array)
    
    print(f"Index generated! Total files: {len(file_mapping)}, Total samples: {len(offsets_array)}")

def load_sample(index_dir: str, global_index: int, full_offsets:list=None, file_mapping:list=None) -> dict:
  
    mapping_path = os.path.join(index_dir, "file_mapping.json")
    offsets_path = os.path.join(index_dir, "offsets.npy")

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


def generate_stage2_mmprojecor_train():
    INDEX_DIR = "/mnt/afs/yangdeyu/GameMLLM/VeOmni-Dev/exp_data/0518_stage2_mm_projector"  # 替换成你的目录

    os.makedirs(INDEX_DIR, exist_ok=True)

    file_paths =[
        # caption
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/blip3_en_11M.jsonl", 0.2], # sa_webdata
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/densefusion_1M.jsonl", 1.0], # 
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/vl3_caption_detail_4M.jsonl", 0.4], # coyo
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/llava_pretrain_558.jsonl", 1.0], # llava
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/llavamed_train_cap_460k.jsonl", 1.0], # medical
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/blip3_zh_8M.jsonl", 0.65],
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/image_video_pretrain/Pixmo/pixmo-cap/train_conv.jsonl", 1.0],

        # ocr/gui
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/image_video_pretrain/ocr/systh-en.jsonl", 1.0], # 68w
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/image_video_pretrain/ocr/synthdog-en.jsonl", 1.0], # 50w
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/image_video_pretrain/ocr/systh-zh.jsonl", 0.3], # 385w
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/image_video_pretrain/ocr/synthdog-zh.jsonl", 1.0], # 50w
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/img_video_instruct/OCR/gameOCR/gameocr_stage1_60w.jsonl",1.0],
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/img_video_instruct/OCR/AnyWord-3M/caption_ocr.jsonl",1.0],
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/img_video_instruct/OCR/AnyWord-3M/zh_ocr_qa_30w.jsonl", 1.0],
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/blip3ocr_200w.jsonl", 0.4], # en scene
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/multiui_caption_150w.json",0.5], # ui
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/LSVT_zh_captions_390k.jsonl", 1.0], # zh scene
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/websight_caption_190w.jsonl", 0.5], # web
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/geo_pretrain_60k.jsonl",1.0], # math ocr
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/screenqa_caption_209k.jsonl",1.0],
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/img_video_instruct/OCR/SynZhOCR/pipelines/data/ja_ocr_userlang.jsonl", 0.5],
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/img_video_instruct/OCR/SynZhOCR/pipelines/data/ja_ocr.jsonl", 0.5],
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/img_video_instruct/OCR/SynZhOCR/pipelines/data/zh_ocr.jsonl", 0.5],
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/img_video_instruct/OCR/SynZhOCR/ocr_train_218w.jsonl", 0.6],

        # video
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/videogpt_webvid_caption_1_5k.jsonl",1.0],
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/img_video_instruct/zh_stage2/webvid.jsonl",1.0], # 40w
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/vript_400k.jsonl", 1.0], # 
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/img_video_instruct/zh_stage2/sharegpt4video_zh.jsonl", 1.0],  # 3w
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/videoufo_zh_detail_337k.jsonl", 1.0],
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/videoufo_train_detailcap_1M.jsonl",0.8],
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/image_video_pretrain/videoufo/train_gemini_caption.jsonl", 0.5],
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/llava178k_caption_172k.jsonl", 1.0],
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/guiworld_caption_21k.jsonl", 1.0],
        # vqa
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/videogpt_webvid_vqa_100k.jsonl", 1.0],
        # motionsight
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/img_video_instruct/MotionSight/train_sft_125k.jsonl", 1.0],

        # 
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/gamebunny_caption_136k.jsonl", 1.0],
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/videogame_qa_74k.jsonl", 1.0],
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/gamecap_gpt4v_70k.jsonl", 1.0],

        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/OCRVQA_stage1.jsonl", 1.0],
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/cambrian_sampled_stage1_177w.jsonl",1.0], # 
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/multiui_vqa_stage1.jsonl", 0.75],
        ]
    
    file_paths = [(path, ratio) for path, ratio in file_paths]

    # get_dataset_statistics(file_paths)
    generate_index_files(file_paths=file_paths, output_dir=INDEX_DIR, regen_offset=True)
    TEST_INDEX = 99                  
    
    try:
        sample = load_sample(INDEX_DIR, TEST_INDEX)
        print("✅ 读取成功！数据预览：")
        # 仅打印前 200 个字符防刷屏
        print(str(sample)) 
    except Exception as e:
        print(f"❌ 读取失败: {e}")


def generate_stage2_audioprojecor_train():
    INDEX_DIR = "/mnt/afs/yangdeyu/GameMLLM/VeOmni-Dev/exp_data/0518_stage2_audio_projector"
    os.makedirs(INDEX_DIR, exist_ok=True)
    root_path = "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/1218_audio_rank96_zhen_aec"

    file_paths = []
    for name in os.listdir(root_path):
        if name.endswith(".jsonl"):
            file_paths.append((os.path.join(root_path, name), 1.0))
    
    generate_index_files(file_paths=file_paths, output_dir=INDEX_DIR, regen_offset=True)
    TEST_INDEX = 100                  
    
    try:
        sample = load_sample(INDEX_DIR, TEST_INDEX)
        print("✅ 读取成功！数据预览：")
        # 仅打印前 200 个字符防刷屏
        print(str(sample)) 
    except Exception as e:
        print(f"❌ 读取失败: {e}")


def generate_stage3_train():

    INDEX_DIR = "/mnt/afs/yangdeyu/GameMLLM/VeOmni-Dev/exp_data/0518_stage3"
    os.makedirs(INDEX_DIR, exist_ok=True)

    filterd_paths = [
        "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/image_video_pretrain/ocr/synthdog-en.jsonl", # 50w 排版非常阴间，会让模型乱看，而且非常模版化
        "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/image_video_pretrain/ocr/systh-zh.jsonl", # 385w 人都看不清，容易让模型产生幻觉，瞎猜
        "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/image_video_pretrain/ocr/synthdog-zh.jsonl", # 50w 排版非常阴间，
        "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/img_video_instruct/OCR/AnyWord-3M/zh_ocr_qa_30w.jsonl",
        "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/multiui_caption_150w.jsonl", # ui
        "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/websight_caption_190w.jsonl", # web
        "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/geo_pretrain_60k.jsonl", # math ocr
        "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/OCRVQA_stage1.jsonl", #回答异常简单
        "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/multiui_vqa_stage1.jsonl", #QA质量差
        "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/videoxl_anomalydet_baaicap_cinepile_ego4d_vcg_vico_700k.jsonl", #没有指定直接回答
        "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/plm_gqa_video_200w.jsonl", #质量较差
        "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/img_video_instruct/VideoVista/videovista_train_exclude_temporal.jsonl", #视频很长，时序性比较差
        "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/OCRVQA_stage2.jsonl", # 回答异常简单，最多只能放在stage1
        "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/plotQA_157k.jsonl", # plot应该是用合成的，全是一个样子的图，有些还渲染错误，答案全是小数点后面N位的，模型根本回答不出来，不可用
        "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/ST_VQA_v3_19k.jsonl", # 没有指定只用一个词回答
        "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/visualgenome_69w.jsonl", # 全是坐标，感觉目前模型学的不好，得系统性测试grounding的能力
        "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/cosyn_train_40w.jsonl", # 图像区分度太低了，渲染的也一般，有重合
        "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/multiui_vqa_stage2.jsonl"
        "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/image_video_pretrain/ocr/unichart_qa/unichart_qa_30w.jsonl", # 坐标系混乱，模版单一，质量较差
        "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/image_video_pretrain/ocr/unichart_table/unichart_table_30w.jsonl", # 数值太不准了，模版较单一，可以重新打标
    
    ]    
    
    all_data_path = [        
        # caption #130w
        ['/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/laion_400k_gpt4v.jsonl', 1.0], # 40w
        ['/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/vl3_caption_detail_4M.jsonl', 0.05], # 20w
        ['/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/image_video_pretrain/ocr/systh-en.jsonl', 0.1], # 6 w
        ['/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/img_video_instruct/OCR/AnyWord-3M/caption_ocr.jsonl', 0.1], # 9w
        ['/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/LSVT_zh_captions_390k.jsonl', 0.1], # 3.9
        ['/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/gamebunny_caption_136k.jsonl', 0.2], # 2.8
        ['/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/gamecap_gemini_70k.jsonl', 1.0], # 7w
        ['/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/densefusion_gpt4v_11w.jsonl', 0.5], # 5.5 
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/image_video_pretrain/Pixmo/pixmo-cap/train_conv.jsonl", 1.0], #  36w
        # image qa # 910w
        ['/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/videogame_qa_74k.jsonl', 1.0], # 7.4w
        ['/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/cambrian_sampled_stage2_355w.jsonl', 1.0], # 355w
        ['/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/textocr_gpt4v_vqa_25k.jsonl', 1.0], # 2.5w
        ['/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/share5o_filter_grounding_139w.jsonl', 0.8], # 128w
        ['/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/image_video_pretrain/ocr/Leopard-Instruct/Leopard_train_107w.jsonl', 0.2], # 21w
        ['/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/rctw_zh_qa_5k.jsonl', 1.0], # 5k
        ['/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/img_video_instruct/OCR/AnyWord-3M/zh_ocr_qa_60w.jsonl', 1.0], # 60w
        ['/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/img_video_instruct/OCR/ZhEn-latex-ocr/train_150k.jsonl', 1.0], # 15w
        ['/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/image_video_pretrain/ocr/systh-zh.jsonl', 0.1], # 6.8w
        ['/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/img_video_instruct/OCR/SynZhOCR/ocr_train_50w.jsonl', 0.5], # 25w
        ['/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/img_video_instruct/OCR/SynZhOCR/ocr_train_218w.jsonl', 0.3], # 60w
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/img_video_instruct/OCR/SynZhOCR/pipelines/data/ja_ocr_userlang.jsonl", 1.0], # 6w
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/img_video_instruct/OCR/SynZhOCR/pipelines/data/ja_ocr.jsonl", 0.5], # 32w
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/img_video_instruct/OCR/SynZhOCR/pipelines/data/zh_ocr.jsonl", 0.4], # 32w
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/img_video_instruct/OCR/gameOCR/gameocr_stage2_90w.jsonl", 1.0], # 90w
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/image_video_pretrain/Pixmo/pixmo-cap-qa/train_merge.jsonl", 1.0],# 11w
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/multiui_vqa_stage2.jsonl", 0.2], # 30w
        # reasoning 
        ['/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/img_video_instruct/OCR/OCRReasoningBench/train_1k.jsonl', 1.0], # 1k
        ['/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/img_video_instruct/OmniAlign-V/train_206k.jsonl', 1.0], # 20w
        ['/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/img_video_instruct/MM-HELIX-100K/train_100k.jsonl', 1.0], # 10w

        # video cap/qa # 416w
        ['/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/img_video_instruct/zh_stage2/detail_videogpt4o_sharegpt4video.jsonl', 1.0], # 4w
        ['/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/img_video_instruct/zh_stage2/options_sthv2_clevererqa_mc_nextqa_kinetics_tgifqa.jsonl', 1.0], # 23.4w
        ['/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/img_video_instruct/zh_stage2/youcook2_textvr_charades_charadesEgo_tacos_tgifcap_qa_hmdb_coin.jsonl', 1.0], # 29w
        ['/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/motionbench_train.jsonl', 1.0], #  5k
        ['/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/llava178k_qa_118w.jsonl', 1.0], # 118w
        ['/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/sharegemini_kinetic_220k.jsonl', 1.0], # 22w
        ['/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/sharegemini_webvid_101k.jsonl', 1.0], # 10w
        ['/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/Taiser_513k_except_Ego4D.jsonl', 1.0], # 51w
        ['/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/Taiser_Ego4D_caption_44k.jsonl', 1.0], # 4.4w
        ['/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/vcg_plus_112k.jsonl', 1.0], # 11w
        ['/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/guiworld_qa_31k.jsonl', 1.0], # 3w
        ['/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/img_video_instruct/Video-R1-data/train_filter_165k_addthink.jsonl', 1.0], # 16.5w
        ['/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/img_video_instruct/videoRFT/train_102k.jsonl', 1.0], # 10w
        ['/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/img_video_instruct/Anomaly_Understanding/train_filter_time_3k.jsonl', 1.0], # 3k
        ['/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/img_video_instruct/MotionSight/train_sft_125k.jsonl', 1.0], # 12.5w
        ['/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/img_video_instruct/VidBridge-R1_training_data/train_filter_time_8k.jsonl', 1.0], # 8k
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/plm_gqa_video_200w.jsonl", 0.5], # 100w
        
        # audio-image/ audio-video / audio-text instruct 550w
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/img_video_instruct/audio_instruct/filterd_audioinstruct_merged_446w.jsonl", 0.8],
        ['/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/img_video_instruct/AudioGenrate/data/game_audioins_0503.jsonl', 0.8],

        # audio asr/aec 380w
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/audio/Voices-in-the-Wild-2M/train.jsonl", 1.0], # 50w
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/audio/pretrain_stage1/raw/train_aishell2_removetest.jsonl", 0.2], # 12w
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/audio/pretrain_stage1/raw/train_commonvoice17_zh.jsonl", 0.2], # 12w
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/audio/pretrain_stage1/raw/magic_primewords_STCMDS.jsonl", 0.2], # 14w
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/audio/pretrain_stage1/raw/filter_emila_zh_0.jsonl", 0.1], # 10w
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/audio/pretrain_stage1/raw/filter_emila_zh_10.jsonl", 0.1], # 10w
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/audio/pretrain_stage1/raw/filter_emila_en_0.jsonl", 0.1], # 10w
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/audio/pretrain_stage1/raw/filter_emila_en_1.jsonl", 0.1], # 10w
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/audio/pretrain_stage1/raw/filter_emila_en_2.jsonl", 0.1], # 10w
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/audio/pretrain_stage1/raw/filter_emila_en_3.jsonl", 0.1], # 10w
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/audio/pretrain_stage1/raw/filter_emila_en_4.jsonl", 0.1], # 10w
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/audio/pretrain_stage1/raw/commonvoice_train_en.jsonl", 0.2],  # 56w
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/audio/pretrain_stage1/raw/librispeech_train_conv.jsonl", 0.5], # 11w
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/audio/pretrain_stage1/raw/giga_conv_0.jsonl", 0.2], # 10w
        ## aec caption/qa
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/audio_aec_caption_607w.jsonl", 0.1], # 60w
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/audio/Audio_AAC_AEC/wavcaps_meld_iemo_compar_audiocaps_test.jsonl",1.0], # 21w
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/audio/Audio_AAC_AEC/AudioSet-Audio-Instructions/translated_0.jsonl", 0.5], # 16w
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/audio/Audio_AAC_AEC/audioset_cla_label_des_train/train_test_instruct.jsonl", 0.3], # 20w
        ["/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/audio/Audio_AAC_AEC/audioset_cla_label_des_train/train_detail_qa.jsonl", 1.0], # 28w

        # text 397w
        ['/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/common_instruct/0724_en_all_329w.jsonl', 0.3], # 96w
        ['/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/common_instruct/0724_zh_all_494w.jsonl', 0.2], # 98w
        ['/mnt/afs/zhouhang/data/wiki_data/bili_wiki/0410_bili_wiki--0410gather_oneAI.jsonl', 0.1], # 40w
        ['/mnt/afs/zhouhang/data/wiki_data/huiji_wiki/0410_huiji_wiki--0410gather_oneAI.jsonl', 0.1], # 14w
        ['/mnt/afs/zhouhang/data/wiki_data/video_md/train_data/4_29.jsonl', 0.2], # 10w 
        ['/mnt/afs/zhouhang/data/wiki_data/youmin_all/train_data/0429--0429_qagather_oneAI.jsonl', 0.2], # 36w
        ['/mnt/afs/zhouhang/data/wiki_data/youmin_youxia/0410_new_filter_merge--0410gather_oneAI.jsonl', 0.2], # 52w
        ['/mnt/afs/zhouhang/data/wiki_data/igdb/hand_data/3_6_hand_igdb.jsonl', 0.5], # 3w
        ['/mnt/afs/zhouhang/data/wiki_data/igdb/3_6_igdb_3w_gather_fix.jsonl', 0.5], # 11w
        ['/mnt/afs/zhouhang/data/wiki_data/wiki_zh/3_9_hand_data/3_9_150w_wiki_markdown.jsonl', 0.2], # 30w
        ['/mnt/afs/jiayi/data/mm_data/game_new/train_data/20260217_lol_add_text_check.jsonl', 0.5] # 9w
        ]
    
    file_paths = [(path, ratio) for path, ratio in all_data_path]
    INDEX_DIR = "/mnt/afs/yangdeyu/GameMLLM/VeOmni-Dev/exp_data/stage3_all_data"
    os.makedirs(INDEX_DIR, exist_ok=True)
    # get_dataset_statistics(file_paths)
    generate_index_files(file_paths=file_paths, output_dir=INDEX_DIR, regen_offset=True)
    TEST_INDEX = 99                  
    
    try:
        sample = load_sample(INDEX_DIR, TEST_INDEX)
        print("✅ 读取成功！数据预览：")
        # 仅打印前 200 个字符防刷屏
        print(str(sample)) 
    except Exception as e:
        print(f"❌ 读取失败: {e}")
     

def generate_mmdata():
    root_path = "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/pretrain_data/stage2_1024_rank96"
    file_paths = []
    for p in os.listdir(root_path):
        if p.endswith(".jsonl"):
           path = os.path.join(root_path, p)
           file_paths.append((path, 1.0))
    INDEX_DIR = "/mnt/afs/yangdeyu/GameMLLM/VeOmni-Dev/exp_data/mmdata_1024_rank96"
    os.makedirs(INDEX_DIR, exist_ok=True)
    generate_index_files(file_paths=file_paths, output_dir=INDEX_DIR, regen_offset=False)
    TEST_INDEX = 99                  
    
    try:
        sample = load_sample(INDEX_DIR, TEST_INDEX)
        print("✅ 读取成功！数据预览：")
        # 仅打印前 200 个字符防刷屏
        print(str(sample)) 
    except Exception as e:
        print(f"❌ 读取失败: {e}")
    


def merge_test():

    paths = [
             "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/data/img_video_instruct/OCR/SynZhOCR/pipelines/data/ja_ocr_testset.jsonl",
             "/mnt/afs/yangdeyu/GameMLLM/LLaVA_hub/exp_data/wukong_mvbench_mvaudio_text_ocr_generalbench_audioaec.json",
             "/mnt/afs/yangdeyu/GameMLLM/VeOmni-Dev/exp_data/text_eval.json"
             ]
    all_data = []
    for name in paths:
        for line in stream_read_json_data(name):
            if "wukong_mvbench_mvaudio_text_ocr_generalbench_audioaec" in name and "text" in line["category"]:
                continue
            else:
                all_data.append(line)

    random.shuffle(all_data)

    save_json_data(data=all_data, file_path="/mnt/afs/yangdeyu/GameMLLM/VeOmni-Dev/exp_data/mvbench_text_ocr_generalbench_audioaec.json")


index_dir = "/mnt/afs/yangdeyu/GameMLLM/VeOmni-Dev/exp_data/0518_stage2_mm_projector"

def worker_task(task_ranges, queue, report_interval=500):
    """
    子进程的核心逻辑：
    task_ranges 是该进程需要处理的索引区间列表，例如 [(0, 10000), (10000, 20000)...]
    """
    local_processed = 0
    local_failed = 0
    mapping_path = os.path.join(index_dir, "file_mapping.json")
    offsets_path = os.path.join(index_dir, "offsets.npy")

   
    with open(mapping_path, "r", encoding="utf-8") as f:
        file_mapping = json.load(f)
    full_offsets = np.load(offsets_path, mmap_mode='r')
    
    for start_idx, end_idx in task_ranges:
        for i in range(start_idx, end_idx):
            try:
                load_sample(index_dir, i, full_offsets, file_mapping)
            except Exception:
                local_failed += 1
            
            local_processed += 1
            
            # 攒够 report_interval 数量汇报一次（兼顾性能与进度条平滑度）
            if local_processed >= report_interval:
                queue.put((local_processed, local_failed))
                local_processed = 0
                local_failed = 0
                
    # 进程退出前，把剩余的尾数汇报上去
    if local_processed > 0 or local_failed > 0:
        queue.put((local_processed, local_failed))



def statistical_failed_data():
    import multiprocessing
    offsets_path = os.path.join(index_dir, "offsets.npy")
    full_offsets = np.load(offsets_path, mmap_mode='r')
    total_samples = len(full_offsets)
    
    num_workers = 10
    queue = multiprocessing.Queue()
    
    print(f"🚀 启动 {num_workers} 个后台进程，数据总量: {total_samples}")

  
    chunk_size = 50000
    ranges = []
    for i in range(0, total_samples, chunk_size):
        ranges.append((i, min(i + chunk_size, total_samples)))
        
    worker_tasks = [[] for _ in range(num_workers)]
    for i, r in enumerate(ranges):
        worker_tasks[i % num_workers].append(r)
        
    # --- 2. 启动子进程 ---
    processes = []
    for i in range(num_workers):
      
        p = multiprocessing.Process(target=worker_task, args=(worker_tasks[i], queue, 200))
        p.start()
        processes.append(p)
        
    # --- 3. 主进程监听队列并丝滑刷新进度条 ---
    total_failed = 0
    completed = 0
    
    # 进度条会实时更新，并且在后缀显示实时的 Failed 数量
    with tqdm(total=total_samples, desc="Checking data", unit=" sample") as pbar:
        while completed < total_samples:
            # 阻塞等待子进程汇报进度
            processed, failed = queue.get()
            
            completed += processed
            total_failed += failed
            
            pbar.update(processed)
       
            if total_failed > 0:
                pbar.set_postfix({"Failed": total_failed})
            
    for p in processes:
        p.join()
        
    print(f"\n✅ 检查完成！")
    print(f"📄 总数据量: {total_samples}")
    print(f"❌ 失败数量: {total_failed}")
    print(f"📉 错误率: {(total_failed/total_samples)*100:.4f}%" if total_samples > 0 else "错误率: N/A")








if __name__ == '__main__':

    # generate_stage2_mmprojecor_train()
    # statistical_failed_data()
    # merge_test()
    # generate_stage3_train()
    generate_mmdata()