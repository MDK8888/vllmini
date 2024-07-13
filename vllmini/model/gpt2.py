import torch
from torch import nn
from transformers import GPT2Config
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
    ) -> torch.Tensor:
        qkv = self.c_attn(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_heads, self.head_dim)
        v = v.view(-1, self.num_heads, self.head_dim)

        attn_output = paged_attention_v1(
            out=torch.empty_like(q),
            query=q,
            key_cache=k,
            value_cache=v,
            num_kv_heads=self.num_heads,
            scale=self.scale,
            block_tables=block_tables,
            seq_lens=seq_lens,
            block_size=64,  # Adjust this value as needed
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

        attn_output = attn_output.contiguous().view(-1, self.hidden_size)
        attn_output = self.c_proj(attn_output)
        return attn_output

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
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(inner_dim, config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
        )
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states
        return hidden_states

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
        kv_caches: List[torch.Tensor],
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
    ) -> torch.Tensor:
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        for i, layer in enumerate(self.h):
            hidden_states = layer(
                hidden_states,
                kv_caches[i],
                block_tables,
                seq_lens,
                max_seq_len,
            )

        hidden_states = self.ln_f(hidden_states)
        return hidden_states

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
        kv_caches: List[torch.Tensor],
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
    ) -> torch.Tensor:
        hidden_states = self.transformer(
            input_ids,
            position_ids,
            kv_caches,
            block_tables,
            seq_lens,
            max_seq_len,
        )
        lm_logits = self.lm_head(hidden_states)
        return lm_logits

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        # Simple greedy sampling
        return torch.argmax(logits, dim=-1)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        
        # Initialize KV caches
        kv_caches = [torch.zeros((batch_size, self.config.num_attention_heads, seq_length, self.config.hidden_size // self.config.num_attention_heads), 
                                 device=input_ids.device) 
                     for _ in range(self.config.num_hidden_layers)]

        for _ in range(max_length - seq_length):
            logits = self.forward(input_ids, position_ids, kv_caches, block_tables, seq_lens, max_length)
            next_token = self.sample(logits[:, -1, :])
            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
            position_ids = torch.cat([position_ids, (position_ids[:, -1] + 1).unsqueeze(-1)], dim=-1)
            seq_lens += 1

        return input_ids

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
