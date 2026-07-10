#include "query_cpu.h"

#include <torch/torch.h>

#include <cmath>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "../hashmap/hashmap_cpu.hpp"

at::Tensor hash_query_cpu(const at::Tensor hash_query,
                          const at::Tensor hash_target,
                          const at::Tensor idx_target) {
  TORCH_CHECK(hash_query.device().is_cpu() && hash_target.device().is_cpu() &&
                  idx_target.device().is_cpu(),
              "hash query tensors must be on CPU");
  TORCH_CHECK(hash_query.scalar_type() == at::kLong &&
                  hash_target.scalar_type() == at::kLong &&
                  idx_target.scalar_type() == at::kLong,
              "hash query tensors must use int64 dtype");
  TORCH_CHECK(hash_query.is_contiguous() && hash_target.is_contiguous() &&
                  idx_target.is_contiguous(),
              "hash query tensors must be contiguous");
  TORCH_CHECK(hash_target.numel() == idx_target.numel(),
              "hash_target and idx_target must have the same length");
  int n = hash_target.size(0);
  int n1 = hash_query.size(0);

  std::unordered_map<int64_t, int64_t> hashmap;
  hashmap.reserve(n);
  at::Tensor out = torch::zeros(
      {n1}, at::device(hash_query.device()).dtype(at::ScalarType::Long));
  for (int idx = 0; idx < n; idx++) {
    int64_t key = *(hash_target.data_ptr<int64_t>() + idx);
    int64_t val = *(idx_target.data_ptr<int64_t>() + idx) + 1;
    hashmap[key] = val;
  }
#pragma omp parallel for
  for (int idx = 0; idx < n1; idx++) {
    int64_t key = *(hash_query.data_ptr<int64_t>() + idx);
    auto iter = hashmap.find(key);
    if (iter != hashmap.end()) {
      *(out.data_ptr<int64_t>() + idx) = iter->second;
    }
  }

  return out;
}
