import torch


def init():
    global benchmark, allow_tf32, allow_fp16, device_capability, hash_rsv_ratio
    benchmark = False
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        device_capability = major * 100 + minor * 10
    else:
        device_capability = 0
    allow_tf32 = device_capability >= 800
    allow_fp16 = device_capability >= 750
    hash_rsv_ratio = 2  # default value, reserve 2x original point count for downsampling
