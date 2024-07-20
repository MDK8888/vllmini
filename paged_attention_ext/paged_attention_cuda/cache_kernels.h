#pragma once
#include <torch/extension.h>
#include <vector>

void swap_blocks(torch::Tensor& src, torch::Tensor& dst, const torch::Tensor& block_mapping);

void copy_blocks(std::vector<torch::Tensor> const& key_caches,
                 std::vector<torch::Tensor> const& value_caches,
                 const torch::Tensor& block_mapping);

void reshape_and_cache(torch::Tensor& key, torch::Tensor& value,
                       torch::Tensor& key_cache, torch::Tensor& value_cache,
                       torch::Tensor& slot_mapping,
                       const std::string& kv_cache_dtype, const double kv_scale);

void reshape_and_cache_flash(torch::Tensor& key, torch::Tensor& value,
                             torch::Tensor& k_cache, torch::Tensor& v_cache,
                             torch::Tensor& slot_mapping,
                             const std::string& kv_cache_dtype);

void convert_fp8(torch::Tensor& dst_cache, torch::Tensor& src_cache,
                 const double kv_scale, const std::string& kv_cache_dtype);
