from .models import *

from mmengine.registry import MODELS

from torch_lattice.nn import BatchNorm

MODELS.register_module('TorchLatticeBatchNorm', force=True, module=BatchNorm)