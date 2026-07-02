#pragma once

#include <torch/torch.h>
#include "../hashmap/hashmap_cuda.cuh"

std::vector<at::Tensor> build_mask_from_kmap(int n_points, int n_out_points,
                                             at::Tensor _kmap,
                                             at::Tensor _kmap_sizes);

std::vector<at::Tensor> compact_out_in_map_ordered(at::Tensor _out_in_map);
std::vector<at::Tensor> compact_out_in_map_fod(at::Tensor _out_in_map);
std::vector<at::Tensor> generative_add_compress_cuda(
    at::Tensor _a_feats, at::Tensor _a_coords,
    at::Tensor _b_feats, at::Tensor _b_coords,
    at::Tensor _matches);
std::vector<at::Tensor> sparse_crop_cuda(
    at::Tensor _feats, at::Tensor _coords,
    at::Tensor _coords_min, at::Tensor _coords_max,
    bool has_min, bool has_max);

std::vector<at::Tensor> build_kernel_map_subm_hashmap(
    hashtable& table,
    at::Tensor _in_coords, at::Tensor _coords_min, at::Tensor _coords_max,
    at::Tensor _kernel_sizes, at::Tensor _stride,
    at::Tensor padding, bool to_insert);

std::vector<at::Tensor> build_kernel_map_subm_hashmap_compact(
    hashtable& table,
    at::Tensor _in_coords, at::Tensor _coords_min, at::Tensor _coords_max,
    at::Tensor _kernel_sizes, at::Tensor _active_kernel_offsets,
    at::Tensor _stride, at::Tensor padding, bool to_insert);

std::vector<at::Tensor> build_kernel_map_downsample_hashmap(
    hashtable& table,
    at::Tensor _in_coords, at::Tensor _coords_min, at::Tensor _coords_max,
    at::Tensor _kernel_sizes, at::Tensor _stride,
    at::Tensor _padding, bool to_insert);

std::vector<at::Tensor> build_kernel_map_subm_hashmap_int32(
    hashtable32& table,
    at::Tensor _in_coords, at::Tensor _coords_min, at::Tensor _coords_max,
    at::Tensor _kernel_sizes, at::Tensor _stride,
    at::Tensor padding, bool to_insert);

std::vector<at::Tensor> build_kernel_map_subm_hashmap_compact_int32(
    hashtable32& table,
    at::Tensor _in_coords, at::Tensor _coords_min, at::Tensor _coords_max,
    at::Tensor _kernel_sizes, at::Tensor _active_kernel_offsets,
    at::Tensor _stride, at::Tensor padding, bool to_insert);

std::vector<at::Tensor> build_kernel_map_downsample_hashmap_int32(
    hashtable32& table,
    at::Tensor _in_coords, at::Tensor _coords_min, at::Tensor _coords_max,
    at::Tensor _kernel_sizes, at::Tensor _stride,
    at::Tensor _padding, bool to_insert);
