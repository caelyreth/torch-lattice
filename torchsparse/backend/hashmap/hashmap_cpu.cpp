#include "hashmap_cpu.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

void HashTableCPU::lookup_vals(const int64_t* const keys,
                               int64_t* const results, const int n) {
#pragma omp parallel for
  for (int idx = 0; idx < n; idx++) {
    int64_t key = keys[idx];
    auto iter = hashmap.find(key);
    if (iter != hashmap.end()) {
      results[idx] = iter->second;
    } else {
      results[idx] = 0;
    }
  }
}

void HashTableCPU::insert_vals(const int64_t* const keys,
                               const int64_t* const vals, const int n) {
  hashmap.reserve(hashmap.size() + n);
  for (int idx = 0; idx < n; idx++) {
    hashmap[keys[idx]] = vals[idx];
  }
}
