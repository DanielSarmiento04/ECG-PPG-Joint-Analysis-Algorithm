import torch

def get_device() -> torch.device:
    """
        get device suppoprt cudo if available, mps for macs with m1/m2 chips else cpu

    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
    
