#pragma once

#include <torch/torch.h>

at::Tensor downsample_cuda(at::Tensor _in_coords, at::Tensor _coords_max,
                           at::Tensor _coords_min, at::Tensor _kernel_sizes,
                           at::Tensor _stride, at::Tensor _padding);

at::Tensor downsample_simple_cuda(at::Tensor _in_coords,
                                  at::Tensor _coords_max,
                                  at::Tensor _stride);

at::Tensor upsample_generative_cuda(at::Tensor _in_coords,
                                    at::Tensor _kernel_offsets,
                                    at::Tensor _stride);
