import random
import numpy as np
import torch

def set_seed(seed, use_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_gpu:
        torch.cuda.manual_seed_all(seed)