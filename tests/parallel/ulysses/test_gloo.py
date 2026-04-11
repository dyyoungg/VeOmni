import os
import torch
import torch.distributed as dist
import datetime

def main():
    # 1. 初始化全局进程组
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=30))
    
    global_rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    
    print(f"[Rank {global_rank}] 全局 NCCL 初始化成功，等待其他节点...")
    dist.barrier()

    unique_groups = [[0, 1], [2, 3], [4, 5], [6, 7]]
    
    my_gloo_group = None
    
    for g_ranks in unique_groups:
        # 所有 8 个进程都要参与创建这 4 个组
        gloo_group = dist.new_group(
            ranks=g_ranks, 
            backend="gloo", 
            timeout=datetime.timedelta(seconds=15)
        )
        # 只有属于这个组的进程，才把句柄保存下来后续使用
        if global_rank in g_ranks:
            my_gloo_group = gloo_group
            
    dist.barrier()
    
    # 3. 模拟 CPU 数据收集
    my_data = torch.tensor([global_rank], dtype=torch.long)
    gather_list = [torch.zeros_like(my_data) for _ in range(2)]
    
    print(f"[Rank {global_rank}] 准备发起 CPU Gloo all_gather...")
    try:
        dist.all_gather(gather_list, my_data, group=my_gloo_group)
        print(f"✅ [Rank {global_rank}] 测试成功! 收集到的数据: {[t.item() for t in gather_list]}")
    except Exception as e:
        print(f"❌ [Rank {global_rank}] Gloo 通信失败: {e}")
        
    dist.destroy_process_group()

if __name__ == "__main__":
    main()