# /Users/jiangpf/Documents/Code/AI/TRIDENT/trident/utils.py

import gc
import torch
from typing import Union

from functools import wraps


def get_device(device: Union[str, torch.device, None] = None) -> torch.device:
    """
    智能选择或验证计算设备。

    选择顺序:
    1. 用户指定的设备 (如果提供)。
    2. 如果未指定，则自动选择最佳可用设备 (CUDA -> MPS -> CPU)。

    Args:
        device: 可以是设备字符串 (如 'cuda:0', 'mps', 'cpu') 或 torch.device 对象。
                如果为 None，则自动选择。

    Returns:
        一个 torch.device 对象。
    """
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        return torch.device(device)

    # 如果 device is None，自动检测
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Apple Silicon (M1/M2/M3) GPU
        return torch.device("mps")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def empty_cache():
    """跨平台清理 PyTorch 内存缓存。"""
    gc.collect()
    if torch.cuda.is_available():
        # CUDA 或 ROCm 环境
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS (Apple Silicon) 没有 empty_cache，就只依赖 gc
        pass
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        # Intel XPU 目前没有 empty_cache
        pass
    else:
        # CPU 平台，没有额外缓存机制
        pass


def autocast_fp16():
    """
    跨平台统一使用 float16 的 autocast 装饰器。
    忽略 TPU，只考虑 CUDA / MPS / XPU / CPU。
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if torch.cuda.is_available():
                device_type = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device_type = "mps"
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                device_type = "xpu"
            else:
                device_type = "cpu"

            with torch.autocast(device_type=device_type, dtype=torch.float16):
                return func(*args, **kwargs)

        return wrapper

    return decorator
