from __future__ import annotations

GROUPS = (
    'tensor',
    'hash',
    'dense',
    'kmap',
    'conv',
    'nn',
    'train',
)

MODES = ('cold_op', 'hot_op', 'backward')

PRESETS = ('smoke', 'standard', 'full')

__all__ = ['GROUPS', 'MODES', 'PRESETS']
