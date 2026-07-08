import numpy as np
import pytest
import torch

import torch_lattice
import torch_lattice.nn as spnn
from torch_lattice.nn import functional as F
from torch_lattice.nn.functional.hash import sphash

from .test_utils import generate_feature_map


def test_generate_feature_map_returns_batch_first_coords():
    sparse = generate_feature_map((3, 4, 5), [6, 6], 2, dtype=np.float32)
    coords = sparse["coords"]

    assert coords.shape[1] == 4
    assert set(coords[:, 0].tolist()) == {0, 1}
    assert coords[:, 1].max() < 3
    assert coords[:, 2].max() < 4
    assert coords[:, 3].max() < 5


def test_spcrop_uses_spatial_xyz_not_batch_column():
    coords = torch.tensor(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 2],
            [0, 2, 1, 1],
        ],
        dtype=torch.int,
        device="cuda",
    )
    feats = torch.arange(coords.size(0), dtype=torch.float32, device="cuda").view(-1, 1)
    tensor = torch_lattice.SparseTensor(feats=feats, coords=coords, spatial_range=(2, 3, 2, 3))

    cropped = F.spcrop(tensor, coords_min=(0, 0, 1), coords_max=(2, 2, 3))

    expected = torch.tensor(
        [
            [1, 0, 0, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 2],
        ],
        dtype=torch.int,
        device="cuda",
    )
    assert torch.equal(cropped.coords, expected)
    assert cropped.spatial_range == tensor.spatial_range

    min_only = F.spcrop(tensor, coords_min=(1, 1, 1))
    expected_min_only = torch.tensor(
        [
            [0, 1, 1, 1],
            [1, 1, 1, 2],
            [0, 2, 1, 1],
        ],
        dtype=torch.int,
        device="cuda",
    )
    assert torch.equal(min_only.coords, expected_min_only)

    max_only = F.spcrop(tensor, coords_max=(1, 1, 2))
    expected_max_only = torch.tensor(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 1],
        ],
        dtype=torch.int,
        device="cuda",
    )
    assert torch.equal(max_only.coords, expected_max_only)


def test_kernel_hash_offsets_apply_to_xyz_on_cpu_and_cuda():
    coords_cpu = torch.tensor(
        [[3, 10, 20, 30], [4, 11, 21, 31]],
        dtype=torch.int,
    )
    offsets_cpu = torch.tensor([[1, 2, 3], [-1, 0, 1]], dtype=torch.int)

    cpu_hash = sphash(coords_cpu, offsets_cpu)
    cuda_hash = sphash(coords_cpu.cuda(), offsets_cpu.cuda()).cpu()

    expected_coords = torch.stack(
        [
            torch.tensor([[3, 11, 22, 33], [4, 12, 23, 34]], dtype=torch.int),
            torch.tensor([[3, 9, 20, 31], [4, 10, 21, 32]], dtype=torch.int),
        ],
        dim=0,
    )
    expected = torch.stack([sphash(expected_coords[i]) for i in range(offsets_cpu.size(0))])

    assert torch.equal(cpu_hash, expected)
    assert torch.equal(cuda_hash, expected)


@pytest.mark.parametrize("module_cls", [spnn.ToBEVReduction, spnn.ToBEVConvolution])
def test_bev_modules_default_to_z_coordinate_dim(module_cls):
    if module_cls is spnn.ToBEVReduction:
        module = module_cls().cuda()
    else:
        module = module_cls(1, 1, n_kernels=4).cuda()
        module.kernel.data.fill_(1)

    coords = torch.tensor(
        [[0, 1, 2, 3], [0, 1, 2, 1], [0, 2, 2, 3]],
        dtype=torch.int,
        device="cuda",
    )
    feats = torch.ones((coords.size(0), 1), dtype=torch.float32, device="cuda")
    tensor = torch_lattice.SparseTensor(feats=feats, coords=coords, spatial_range=(1, 3, 3, 4))

    out = module(tensor)

    assert torch.all(out.coords[:, 3] == 0)


def test_bev_height_compression_default_uses_z_coordinate_dim():
    module = spnn.ToBEVHeightCompression(1, shape=(3, 4, 5)).cuda()
    coords = torch.tensor(
        [[0, 1, 2, 3], [0, 1, 2, 4], [0, 2, 1, 0]],
        dtype=torch.int,
        device="cuda",
    )
    feats = torch.ones((coords.size(0), 1), dtype=torch.float32, device="cuda")
    tensor = torch_lattice.SparseTensor(feats=feats, coords=coords, spatial_range=(1, 3, 4, 5))

    out = module(tensor)

    assert out.shape == (1, 5, 3, 4)
    assert out[0, 3, 1, 2] == 1
    assert out[0, 4, 1, 2] == 1
    assert out[0, 0, 2, 1] == 1


def test_compact_on_the_fly_kmap_uses_int32_hashmap():
    config = F.conv_config.get_default_conv_config().copy()
    config.dataflow = F.Dataflow.ImplicitGEMM
    config.kmap_mode = "hashmap_on_the_fly"
    config.ifsort = False
    F.conv_config.set_global_conv_config(config)

    coords = torch.tensor(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]],
        dtype=torch.int,
        device="cuda",
    )
    feats = torch.randn((coords.size(0), 4), dtype=torch.float16, device="cuda")
    tensor = torch_lattice.SparseTensor(feats=feats, coords=coords, spatial_range=(1, 4, 1, 1))

    try:
        spnn.SubmConv3d(4, 4, kernel_size=3, bias=False).cuda().half()(tensor)
        hashmap_keys, _ = tensor._caches.hashmaps[(1, 1, 1)]
    finally:
        F.conv_config.clear_global_conv_config()

    assert hashmap_keys.dtype == torch.int32


def test_wide_coordinate_stride2_conv_uses_int64_kmap_safely():
    config = F.conv_config.get_default_conv_config().copy()
    config.dataflow = F.Dataflow.ImplicitGEMM
    config.kmap_mode = "hashmap_on_the_fly"
    config.ifsort = True
    config.split_mask_num = 2
    F.conv_config.set_global_conv_config(config)

    coords = torch.tensor(
        [
            [0, 10_199_980, 11_400_068, 13_800_188],
            [0, 10_199_981, 11_400_068, 13_800_188],
            [0, 10_199_980, 11_400_069, 13_800_188],
            [0, 10_199_981, 11_400_069, 13_800_188],
        ],
        dtype=torch.int,
        device="cuda",
    )
    feats = torch.randn((coords.size(0), 4), dtype=torch.float16, device="cuda")
    tensor = torch_lattice.SparseTensor(
        feats=feats,
        coords=coords,
        spatial_range=(1, 10_199_984, 11_400_071, 13_800_191),
    )

    try:
        out = spnn.Conv3d(4, 4, kernel_size=2, stride=2, bias=False).cuda().half()(tensor)
        torch.cuda.synchronize()
        hashmap_keys, _ = tensor._caches.hashmaps[(2, 2, 2)]
    finally:
        F.conv_config.clear_global_conv_config()

    assert hashmap_keys.dtype == torch.int64
    assert out.feats.numel() > 0


def test_fetch_on_demand_fused_falls_back_for_large_quantified_map():
    config = F.conv_config.get_default_conv_config().copy()
    config.dataflow = F.Dataflow.FetchOnDemand
    config.kmap_mode = "hashmap_on_the_fly"
    config.FOD_fusion = True
    F.conv_config.set_global_conv_config(config)

    axis = torch.arange(64, dtype=torch.int, device="cuda")
    z, y, x = torch.meshgrid(axis, axis, axis, indexing="ij")
    coords = torch.stack(
        [torch.zeros_like(x.reshape(-1)), x.reshape(-1), y.reshape(-1), z.reshape(-1)],
        dim=1,
    ).contiguous()
    feats = torch.randn((coords.size(0), 32), dtype=torch.float16, device="cuda")
    tensor = torch_lattice.SparseTensor(feats=feats, coords=coords, spatial_range=(1, 64, 64, 64))

    try:
        out = spnn.SubmConv3d(32, 32, kernel_size=3, bias=False).cuda().half()(tensor)
        torch.cuda.synchronize()
    finally:
        F.conv_config.clear_global_conv_config()

    assert out.feats.shape == feats.shape


def test_kernel_map_cache_is_separated_by_dataflow():
    coords = torch.tensor(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]],
        dtype=torch.int,
        device="cuda",
    )
    feats = torch.randn((coords.size(0), 4), dtype=torch.float16, device="cuda")
    tensor = torch_lattice.SparseTensor(feats=feats, coords=coords, spatial_range=(1, 4, 1, 1))
    conv = spnn.SubmConv3d(4, 4, kernel_size=3, bias=False).cuda().half()

    config = F.conv_config.get_default_conv_config().copy()
    config.kmap_mode = "hashmap_on_the_fly"

    try:
        config.dataflow = F.Dataflow.ImplicitGEMM
        F.conv_config.set_global_conv_config(config)
        conv(tensor)

        config.dataflow = F.Dataflow.FetchOnDemand
        config.FOD_fusion = False
        F.conv_config.set_global_conv_config(config)
        conv(tensor)
        torch.cuda.synchronize()
    finally:
        F.conv_config.clear_global_conv_config()

    assert len(tensor._caches.kmaps) == 2
    assert all(kmap.get("qmapsize") is None for kmap in tensor._caches.kmaps.values())


def test_fetch_on_demand_fusion_flag_controls_quantified_kmap_metadata():
    coords = torch.tensor(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]],
        dtype=torch.int,
        device="cuda",
    )
    feats = torch.randn((coords.size(0), 4), dtype=torch.float16, device="cuda")
    conv = spnn.SubmConv3d(4, 4, kernel_size=3, bias=False).cuda().half()
    config = F.conv_config.get_default_conv_config().copy()
    config.dataflow = F.Dataflow.FetchOnDemand
    config.kmap_mode = "hashmap_on_the_fly"

    try:
        config.FOD_fusion = False
        F.conv_config.set_global_conv_config(config)
        no_fusion = torch_lattice.SparseTensor(
            feats=feats, coords=coords, spatial_range=(1, 4, 1, 1)
        )
        conv(no_fusion)
        torch.cuda.synchronize()
        no_fusion_kmap = next(iter(no_fusion._caches.kmaps.values()))

        config.FOD_fusion = True
        F.conv_config.set_global_conv_config(config)
        fused = torch_lattice.SparseTensor(
            feats=feats, coords=coords, spatial_range=(1, 4, 1, 1)
        )
        conv(fused)
        torch.cuda.synchronize()
        fused_kmap = next(iter(fused._caches.kmaps.values()))
    finally:
        F.conv_config.clear_global_conv_config()

    assert no_fusion_kmap.get("qmapsize") is None
    assert no_fusion_kmap.get("qnbaddrs") is None
    assert fused_kmap.get("qmapsize") is not None
    assert fused_kmap.get("qnbaddrs") is not None
