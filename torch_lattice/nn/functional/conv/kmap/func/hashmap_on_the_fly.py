from typing import Dict, Tuple, Optional
import torch

import torch_lattice.backend
import torch_lattice.backends
from torch_lattice.nn.functional.conv.kmap.layout import (
    set_fod_neighbor_maps,
    set_neighbor_pairs,
)
from torch_lattice.utils import make_tensor


_INT32_MAX_SAFE_KEY = 2**31 - 2


def _can_use_int32_hashmap(spatial_range: Optional[Tuple[int]]) -> bool:
    if spatial_range is None:
        return False

    key_space = 1
    for extent in spatial_range:
        key_space *= int(extent)
        if key_space > _INT32_MAX_SAFE_KEY:
            return False
    return True


def build_kmap_implicit_GEMM_hashmap_on_the_fly(
    kmap: Dict,
    input_node_num: int,
    _coords: torch.Tensor,
    kernel_size: torch.Tensor,
    stride: torch.Tensor,
    padding: torch.Tensor,
    spatial_range: Optional[Tuple[int]] = None,
    cta_M: int = 128,
    subm: bool = False,
    ifsort: bool = False,
    split_mask_num: int = 1,
    IGEMM_center_only: bool = False,
    active_kernel_offsets: Optional[torch.Tensor] = None,
) -> Dict:
    kmap["coords"] = _coords
    kmap["spatial_range"] = spatial_range
    coords = _coords.contiguous()
    if spatial_range is not None:
        coords_max_tuple = tuple(x - 1 for x in spatial_range)
        coords_max = make_tensor(
            coords_max_tuple, dtype=torch.int, device=coords.device
        )
    else:
        coords_max = coords.max(0).values
        if not subm:
            coords_max[1:] = (
                coords_max[1:] + 2 * padding - (kernel_size - 1)
            ) // stride

    coords_min = make_tensor((0, 0, 0, 0), dtype=torch.int, device=coords.device)

    use_int32_hashmap = (
        _can_use_int32_hashmap(spatial_range)
        if kmap["hashmap_keys"] is None
        else kmap["hashmap_keys"].dtype == torch.int32
    )
    use_compact_subm = (
        subm
        and active_kernel_offsets is not None
        and int(active_kernel_offsets.numel()) > 0
        and hasattr(torch_lattice.backend, "build_kernel_map_subm_hashmap_compact")
        and hasattr(
            torch_lattice.backend, "build_kernel_map_subm_hashmap_compact_int32"
        )
    )
    if subm:
        if use_compact_subm:
            func = (
                torch_lattice.backend.build_kernel_map_subm_hashmap_compact_int32
                if use_int32_hashmap
                else torch_lattice.backend.build_kernel_map_subm_hashmap_compact
            )
        else:
            func = (
                torch_lattice.backend.build_kernel_map_subm_hashmap_int32
                if use_int32_hashmap
                else torch_lattice.backend.build_kernel_map_subm_hashmap
            )
    else:
        func = (
            torch_lattice.backend.build_kernel_map_downsample_hashmap_int32
            if use_int32_hashmap
            else torch_lattice.backend.build_kernel_map_downsample_hashmap
        )
    to_insert = False

    if torch_lattice.backends.hash_rsv_ratio < 2:
        raise ValueError(
            "hash_rsv_ratio must be at least 2, got "
            f"{torch_lattice.backends.hash_rsv_ratio}"
        )
    hashmap_capacity = max(
        512, int(torch_lattice.backends.hash_rsv_ratio * _coords.shape[0])
    )
    if kmap["hashmap_keys"] is None:
        kmap["hashmap_keys"] = torch.zeros(
            hashmap_capacity,
            dtype=torch.int32 if use_int32_hashmap else torch.int64,
            device=coords.device,
        )
        to_insert = True
    if kmap["hashmap_vals"] is None:
        kmap["hashmap_vals"] = torch.zeros(
            hashmap_capacity, dtype=torch.int32, device=coords.device
        )
    hashmap_cls = (
        torch_lattice.backend.GPUHashTable32
        if kmap["hashmap_keys"].dtype == torch.int32
        else torch_lattice.backend.GPUHashTable
    )
    hashtable = hashmap_cls(kmap["hashmap_keys"], kmap["hashmap_vals"])

    if use_compact_subm:
        out = func(
            hashtable,
            coords,
            coords_min,
            coords_max,
            kernel_size,
            active_kernel_offsets.to(device=coords.device, dtype=torch.int32),
            stride,
            padding,
            to_insert,
        )
    else:
        out = func(
            hashtable,
            coords,
            coords_min,
            coords_max,
            kernel_size,
            stride,
            padding,
            to_insert,
        )

    # update kernel_map
    out_in_map = out[0]
    kmap["out_in_map"] = out_in_map
    if len(out) != 1:
        coords = out[1]
        kmap["coords"] = coords
    kmap["sizes"] = (input_node_num, coords.shape[0])

    if ifsort:
        bitmask = torch_lattice.backend.derive_bitmask_from_out_in_map(
            out_in_map, split_mask_num, kmap["sizes"][1]
        )
        sorted_mask, reorder_loc = torch.sort(bitmask, descending=True)
        reorder_loc = reorder_loc.to(torch.int32)
        reorder_out_in_map = torch_lattice.backend.reorder_out_in_map_cuda(
            out_in_map, reorder_loc
        )
        reduced_sorted_mask = torch_lattice.backend.reduce_bitmask_cuda(
            sorted_mask, cta_M
        )
        kmap["reorder_out_in_map"] = reorder_out_in_map
        kmap["reduced_sorted_mask"] = reduced_sorted_mask
        kmap["reorder_loc"] = reorder_loc
        kmap["sorted_mask"] = sorted_mask

    if IGEMM_center_only and subm and int(torch.prod(kernel_size).item()) % 2 == 1:
        results = torch.t(out_in_map).contiguous()
        nbsizes = torch.sum(results != -1, dim=1).to(torch.int)
        nbsizes_cpu = nbsizes.cpu().contiguous()
        mid_kernel = nbsizes_cpu.numel() // 2
        kmap["IGEMM_center_only"] = int(nbsizes_cpu[mid_kernel]) == int(
            coords.shape[0]
        ) and int(nbsizes_cpu.sum()) == int(coords.shape[0])
    else:
        kmap["IGEMM_center_only"] = False

    return kmap


def build_kmap_Gather_Scatter_hashmap_on_the_fly(
    kmap: Dict,
    input_node_num: int,
    _coords: torch.Tensor,
    kernel_size: torch.Tensor,
    stride: torch.Tensor,
    padding: torch.Tensor,
    spatial_range: Optional[Tuple[int]] = None,
    cta_M: int = 128,
    subm: bool = False,
    active_kernel_offsets: Optional[torch.Tensor] = None,
) -> Dict:

    kmap = build_kmap_implicit_GEMM_hashmap_on_the_fly(
        kmap,
        input_node_num,
        _coords,
        kernel_size,
        stride,
        padding,
        spatial_range,
        cta_M,
        subm,
        False,
        1,
        active_kernel_offsets=active_kernel_offsets,
    )

    if kmap["out_in_map"].is_cuda and hasattr(
        torch_lattice.backend, "compact_out_in_map_ordered"
    ):
        nbmaps, nbsizes, _, _ = torch_lattice.backend.compact_out_in_map_ordered(
            kmap["out_in_map"]
        )
    else:
        results = torch.t(kmap["out_in_map"]).contiguous()
        nbsizes = torch.sum(results != -1, dim=1)
        nbmaps = torch.nonzero(results != -1)
        nbmaps[:, 0] = results.view(-1)[nbmaps[:, 0] * results.size(1) + nbmaps[:, 1]]
        # important for build masks
        nbmaps = nbmaps.contiguous()
    nbmaps = set_neighbor_pairs(kmap, nbmaps)
    input_mask, output_mask = torch_lattice.backend.build_mask_from_kmap(
        _coords.shape[0],
        kmap["coords"].shape[0],
        nbmaps,
        nbsizes.int()[0 : kmap["coords"].shape[0]],
    )

    kmap["nbsizes"] = nbsizes
    kmap["nbsizes_cpu"] = nbsizes.int().cpu().contiguous()
    kmap["input_mask"] = input_mask
    kmap["output_mask"] = output_mask
    kmap["active_kernel_offsets"] = active_kernel_offsets

    return kmap


def build_kmap_Fetch_on_Demand_hashmap_on_the_fly(
    kmap: Dict,
    input_node_num: int,
    _coords: torch.Tensor,
    kernel_size: torch.Tensor,
    stride: torch.Tensor,
    padding: torch.Tensor,
    spatial_range: Optional[Tuple[int]] = None,
    cta_M: int = 128,
    subm: bool = False,
    FOD_fusion: bool = True,
    active_kernel_offsets: Optional[torch.Tensor] = None,
) -> Dict:

    kmap = build_kmap_implicit_GEMM_hashmap_on_the_fly(
        kmap,
        input_node_num,
        _coords,
        kernel_size,
        stride,
        padding,
        spatial_range,
        cta_M,
        subm,
        False,
        1,
        active_kernel_offsets=active_kernel_offsets,
    )

    if (
        kmap["out_in_map"].is_cuda
        and kmap["out_in_map"].size(1) >= 9
        and hasattr(torch_lattice.backend, "compact_out_in_map_fod")
    ):
        fod_map, nbmaps, nbsizes, nbaddrs, qnbaddrs = (
            torch_lattice.backend.compact_out_in_map_fod(kmap["out_in_map"])
        )
    else:
        results = torch.t(kmap["out_in_map"]).contiguous()
        nbsizes = torch.sum(results != -1, dim=1).to(torch.int)
        nbmaps = torch.nonzero(results != -1)
        nbmaps[:, 0] = results.view(-1)[nbmaps[:, 0] * results.size(1) + nbmaps[:, 1]]
        nbmaps = nbmaps.int().contiguous()
        fod_map = nbmaps.t().contiguous()
        if FOD_fusion:
            kernel_volume = nbsizes.size(0)
            nbaddrs = torch.zeros(
                (kernel_volume + 1), dtype=torch.int, device=nbmaps.device
            )
            qnbaddrs = torch.zeros(
                (kernel_volume + 1), dtype=torch.int, device=nbmaps.device
            )

            # Derive quantified arrays
            torch_lattice.backend.exclusive_scan_quantified_wrapper(
                kernel_volume, nbsizes, nbaddrs, qnbaddrs
            )

    set_fod_neighbor_maps(kmap, nbmaps, fod_map)
    kmap["nbsizes"] = nbsizes
    nbsizes_cpu = nbsizes.cpu().contiguous()
    kmap["nbsizes_cpu"] = nbsizes_cpu
    if subm and nbsizes_cpu.numel() % 2 == 1:
        mid_kernel = nbsizes_cpu.numel() // 2
        mapsize = nbmaps.size(0)
        kmap["FOD_center_only"] = int(nbsizes_cpu[mid_kernel]) == int(mapsize) and int(
            nbsizes_cpu.sum()
        ) == int(mapsize)
    else:
        kmap["FOD_center_only"] = False

    if FOD_fusion:
        kmap["nbaddrs"] = nbaddrs
        kmap["qnbaddrs"] = qnbaddrs
        kmap["qmapsize"] = qnbaddrs[-1].cpu().int()
    kmap["active_kernel_offsets"] = active_kernel_offsets

    return kmap
