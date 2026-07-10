#include <torch/torch.h>
#include <c10/cuda/CUDAException.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "../hashmap/hashmap_cuda.cuh"

__global__ void convert_out_in_map_kernel(const int* out_in_map, int* out_in_map_t, int n, int kernel_volume){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= n * kernel_volume) return;
  int input_idx = out_in_map[idx];
  if(input_idx < 0) return;
  out_in_map_t[idx % kernel_volume + input_idx * kernel_volume] = idx / kernel_volume;
}

__global__ void derive_bit_mask_from_out_in_map_kernel(int* out_in_map, int* bitmask, int valid_n, int n, int kernel_volume, int split_mask_num){
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = tidx / split_mask_num;
  if(idx >= valid_n) return;
  int split_mask_iter = tidx % split_mask_num;
  int split_mask_len = (kernel_volume + split_mask_num - 1) / split_mask_num;
  int* cur_out_in_map = out_in_map + kernel_volume * idx + split_mask_iter * split_mask_len;
  if (split_mask_iter == (split_mask_num - 1)) // The last tile
    split_mask_len = kernel_volume - split_mask_iter * split_mask_len;
  int cur_bitmask = 0;
  for(int i = 0; i < split_mask_len; i++){
    cur_bitmask += (int)(cur_out_in_map[i] >= 0) * (int)(1u << i);
  }
  bitmask[split_mask_iter * n + idx] = cur_bitmask;
}

at::Tensor hash_query_cuda(const at::Tensor hash_query,
                           const at::Tensor hash_target,
                           const at::Tensor idx_target) {
  TORCH_CHECK(hash_query.is_cuda() && hash_target.is_cuda() && idx_target.is_cuda(),
              "hash query tensors must be CUDA tensors");
  TORCH_CHECK(hash_query.scalar_type() == at::kLong &&
                  hash_target.scalar_type() == at::kLong,
              "hash_query and hash_target must use int64 dtype");
  TORCH_CHECK(hash_query.is_contiguous() && hash_target.is_contiguous(),
              "hash_query and hash_target must be contiguous");
  // return group_point_forward_gpu(points, indices);
  int n = hash_target.size(0);
  int n1 = hash_query.size(0);
  hashtable in_hash_table(n * 2);

  in_hash_table.insert_many(hash_target.data_ptr<int64_t>(), n);

  at::Tensor out = torch::zeros(
      {n1}, at::device(hash_query.device()).dtype(at::ScalarType::Int));
  in_hash_table.lookup_many(hash_query.data_ptr<int64_t>(), out.data_ptr<int>(), n1);
  return out;
}


void convert_transposed_out_in_map(const at::Tensor out_in_map,
                            at::Tensor out_in_map_t) {
  TORCH_CHECK(out_in_map.is_cuda() && out_in_map_t.is_cuda(),
              "relation tensors must be CUDA tensors");
  TORCH_CHECK(out_in_map.scalar_type() == at::kInt &&
                  out_in_map_t.scalar_type() == at::kInt,
              "relation tensors must use int32 dtype");
  TORCH_CHECK(out_in_map.dim() == 2 && out_in_map_t.dim() == 2,
              "relation tensors must be rank 2");
  TORCH_CHECK(out_in_map.is_contiguous() && out_in_map_t.is_contiguous(),
              "relation tensors must be contiguous");
  if (out_in_map.numel() == 0) return;
  convert_out_in_map_kernel<<<(out_in_map.size(0) * out_in_map.size(1) + 255) / 256, 256>>>(
    out_in_map.data_ptr<int>(), out_in_map_t.data_ptr<int>(), out_in_map.size(0), out_in_map.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}




at::Tensor derive_bitmask_from_out_in_map(const at::Tensor out_in_map, const int split_mask_num, int valid_n) {
  TORCH_CHECK(out_in_map.is_cuda() && out_in_map.scalar_type() == at::kInt,
              "out_in_map must be a CUDA int32 tensor");
  TORCH_CHECK(out_in_map.dim() == 2 && out_in_map.is_contiguous(),
              "out_in_map must be contiguous and rank 2");
  TORCH_CHECK(split_mask_num > 0, "split_mask_num must be positive");
  TORCH_CHECK(valid_n >= 0 && valid_n <= out_in_map.size(0),
              "valid_n must be within out_in_map rows");
  at::Tensor bitmask = torch::full(
      {split_mask_num, out_in_map.size(0)}, -1, at::device(out_in_map.device()).dtype(at::ScalarType::Int));
  if (out_in_map.size(0) == 0) return bitmask;
  derive_bit_mask_from_out_in_map_kernel<<<(split_mask_num * out_in_map.size(0) + 255) / 256, 256>>>(
    out_in_map.data_ptr<int>(), bitmask.data_ptr<int>(), valid_n, out_in_map.size(0), out_in_map.size(1), split_mask_num);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return bitmask;
}
