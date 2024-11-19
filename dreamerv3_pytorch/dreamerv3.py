import torch
from torch.nn import Module, ModuleList

import einx
from einops import rearrange, repeat, pack, unpack

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

class DreamerV3(Module):
    def __init__(self):
        super().__init__()
