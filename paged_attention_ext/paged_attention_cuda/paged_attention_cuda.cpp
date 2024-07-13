#include <torch/extension.h>

#define TORCH_EXTENSION_NAME paged_attention_cuda
// Declare the CUDA functions
void paged_attention_v1(
    torch::Tensor& out,
    torch::Tensor& query,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    int64_t num_kv_heads,
    double scale,
    torch::Tensor& block_tables,
    torch::Tensor& seq_lens,
    int64_t block_size,
    int64_t max_seq_len,
    const c10::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype,
    double kv_scale,
    const int64_t tp_rank,
    const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride,
    const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step);

void paged_attention_v2(
    torch::Tensor& out,
    torch::Tensor& exp_sums,
    torch::Tensor& max_logits,
    torch::Tensor& tmp_out,
    torch::Tensor& query,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    int64_t num_kv_heads,
    double scale,
    torch::Tensor& block_tables,
    torch::Tensor& seq_lens,
    int64_t block_size,
    int64_t max_seq_len,
    const c10::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype,
    double kv_scale,
    const int64_t tp_rank,
    const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride,
    const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step);

// Binding code
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("paged_attention_v1", &paged_attention_v1, "Paged Attention V1");
    m.def("paged_attention_v2", &paged_attention_v2, "Paged Attention V2");
}