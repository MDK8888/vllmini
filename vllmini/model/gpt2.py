import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel
from typing import List, Optional, Tuple, Iterable

from paged_attention_cuda import paged_attention_v1

class GPT2Attention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5

        self.c_attn = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=True)
        self.c_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        # Parameters for paged attention
        self.block_size = 16  # You may want to make this configurable
        self.max_num_blocks_per_seq = 1024  # Adjust as needed

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.size()
        assert seq_length == 1, "Given the kv cache, only 1 hidden state is necessary"
        hidden_states = hidden_states.view(batch_size, -1)
        qkv = self.c_attn(hidden_states)
        q, k, v = qkv.split(self.hidden_size, dim=-1)
        
        q = q.view(batch_size, self.num_heads, self.head_dim)
        k = k.view(batch_size, self.num_heads, -1, self.head_dim)
        v = v.view(batch_size, self.num_heads, -1, self.head_dim)

        # Prepare the caches if they're not already in the correct format
        key_cache = key_cache.view(batch_size, self.num_heads, -1, self.head_dim)
        key_cache = torch.cat([key_cache, k], dim=-2)
        value_cache = value_cache.view(batch_size, self.num_heads, -1, self.head_dim)
        value_cache = torch.cat([value_cache, v], dim=-2)

        # Ensure block_tables has the correct shape
        if block_tables.dim() != 2:
            block_tables = block_tables.view(batch_size, -1)
        
        # Ensure seq_lens has the correct shape
        if seq_lens.dim() != 1:
            seq_lens = seq_lens.view(-1)

        out = torch.empty_like(q)
        paged_attention_v1(
            out=out,
            query=q,
            key_cache=key_cache,
            value_cache=value_cache,
            num_kv_heads=self.num_heads,
            scale=self.scale,
            block_tables=block_tables,
            seq_lens=seq_lens,
            block_size=self.block_size,
            max_seq_len=max_seq_len,
            alibi_slopes=None,
            kv_cache_dtype="float32",
            kv_scale=1.0,
            tp_rank=0,
            blocksparse_local_blocks=0,
            blocksparse_vert_stride=1,
            blocksparse_block_size=1,
            blocksparse_head_sliding_step=0
        )

        attn_output = out.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
        attn_output = self.c_proj(attn_output)
        return attn_output, key_cache, value_cache

class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size: int, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.hidden_size, intermediate_size, bias=True)
        self.c_proj = nn.Linear(intermediate_size, config.hidden_size, bias=True)
        self.act = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return hidden_states

class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config.intermediate_size, config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output, new_key_cache, new_value_cache = self.attn(
            hidden_states,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            max_seq_len
        )
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_output = self.mlp(hidden_states)
        hidden_states = feed_forward_output + residual

        return hidden_states, new_key_cache, new_value_cache

class GPT2Model(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        key_caches: List[torch.Tensor],
        value_caches: List[torch.Tensor],
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        new_key_caches = []
        new_value_caches = []

        for i, layer in enumerate(self.h):
            hidden_states, new_key_cache, new_value_cache = layer(
                hidden_states,
                key_caches[i],
                value_caches[i],
                block_tables,
                seq_lens,
                max_seq_len,
            )
            new_key_caches.append(new_key_cache)
            new_value_caches.append(new_value_cache)

        hidden_states = self.ln_f(hidden_states)
        return hidden_states, new_key_caches, new_value_caches

class GPT2LMHeadModel(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights between the input embedding (wte) and the output embedding (lm_head)
        self.lm_head.weight = self.transformer.wte.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        key_caches: List[torch.Tensor],
        value_caches: List[torch.Tensor],
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        hidden_states, new_key_caches, new_value_caches = self.transformer(
            input_ids,
            position_ids,
            key_caches,
            value_caches,
            block_tables,
            seq_lens,
            max_seq_len,
        )
        lm_logits = self.lm_head(hidden_states)
        return lm_logits, new_key_caches, new_value_caches

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        model_state_dict = self.state_dict()
        for name, loaded_weight in weights:
            if "lm_head.weight" in name:
                # GPT-2 ties the weights of the embedding layer and the final linear layer
                continue
            if ".attn.bias" in name or ".attn.masked_bias" in name:
                # Skip attention mask
                continue
            
            # Remove 'transformer.' prefix if present
            if name.startswith("transformer."):
                name = name[len("transformer."):]
            
            # Handle Conv1D to Linear conversion
            if any(conv1d_name in name for conv1d_name in ["c_attn", "c_proj", "c_fc"]):
                if name.endswith(".weight"):
                    loaded_weight = loaded_weight.t()
            
            # Update the name to match our model's state dict
            name = name.replace("h.", "transformer.h.")
            if name not in model_state_dict:
                print(f"Skipping weight {name} as it's not in the model")
                continue
            
            if model_state_dict[name].shape != loaded_weight.shape:
                print(f"Shape mismatch for {name}: expected {model_state_dict[name].shape}, got {loaded_weight.shape}")
                continue
            
            model_state_dict[name].copy_(loaded_weight)
        
        self.load_state_dict(model_state_dict)
        print("Weights loaded successfully")