import torch

if not torch.cuda.is_available():
    print("CUDA不可用，无法使用GPU设备")
else:
    if torch.cuda.device_count() > 2:
        device = torch.device("cuda:2")
        print(f"总显存: {torch.cuda.get_device_properties(2).total_memory / 1e9:.2f} GB")
        print(f"已用显存: {torch.cuda.memory_allocated(2) / 1e9:.2f} GB")
        print(f"缓存显存: {torch.cuda.memory_reserved(2) / 1e9:.2f} GB")
        print(f"使用设备: {device} ({torch.cuda.get_device_name(device)})")
        
        # 新增：清空缓存
        torch.cuda.empty_cache()
        
        # 然后创建张量
        a = torch.tensor([1.0, 2.0, 3.0], device=device)
        b = torch.tensor([4.0, 5.0, 6.0], device=device)
        c = a + b
        
        print("计算结果:", c)
        print("结果所在设备:", c.device)
    else:
        print(f"不存在cuda:2设备，当前系统共有 {torch.cuda.device_count()} 个CUDA设备")