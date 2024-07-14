import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel
from typing import List, Optional, Tuple
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

        self.block_size = 16  # You may want to make this configurable

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        is_prefill: bool = True,
        key_cache: Optional[torch.Tensor] = None,
        value_cache: Optional[torch.Tensor] = None,
        block_tables: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        max_seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.c_attn(hidden_states)
        query, key, value = qkv.split(self.hidden_size, dim=-1)

        query = query.reshape(batch_size, self.num_heads, seq_len, self.head_dim)
        key = key.reshape(batch_size, self.num_heads, seq_len, self.head_dim)
        value = value.reshape(batch_size, self.num_heads, seq_len, self.head_dim)

        if is_prefill:
            attn_output = self._vanilla_attention(query, key, value, attention_mask)
        else:
            key_cache = torch.cat([key_cache, key], dim=-2)
            value_cache = torch.cat([value_cache, value], dim=-2)
            attn_output = self._paged_attention(query, key_cache, value_cache, block_tables, seq_lens, max_seq_len)

        present = (key, value) if use_cache else None

        attn_output = attn_output.transpose(1, 2).contiguous().view(hidden_states.size())
        attn_output = self.c_proj(attn_output)

        return attn_output, present

    def _vanilla_attention(self, query, key, value, attention_mask):
        attn_weights = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        return torch.matmul(attn_weights, value)

    def _paged_attention(self, query, key_cache, value_cache, block_tables, seq_lens, max_seq_len):
        query = query.squeeze(-2)
        out = torch.empty_like(query)

        paged_attention_v1(
            out,
            query,
            key_cache,
            value_cache,
            self.num_heads,
            self.scale,
            block_tables,
            seq_lens,
            self.block_size,
            max_seq_len,
            None,
            "auto",
            1.0,
            0,
            0,
            1,
            1,
            0
        )
        return out.unsqueeze(-2)

class GPT2MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.hidden_size
        self.c_fc = nn.Linear(config.hidden_size, inner_dim, bias=True)
        self.c_proj = nn.Linear(inner_dim, config.hidden_size, bias=True)
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
        self.mlp = GPT2MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        is_prefill: bool = True,
        key_cache: Optional[torch.Tensor] = None,
        value_cache: Optional[torch.Tensor] = None,
        block_tables: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        max_seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output, present = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            is_prefill=is_prefill,
            key_cache=key_cache,
            value_cache=value_cache,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
        )
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_output = self.mlp(hidden_states)
        hidden_states = feed_forward_output + residual

        return hidden_states, present

class GPT2Model(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        is_prefill: bool = True,
        key_caches: Optional[List[torch.Tensor]] = None,
        value_caches: Optional[List[torch.Tensor]] = None,
        block_tables: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        max_seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        hidden_states = self.wte(input_ids) + self.wpe(position_ids)

        presents = [] if use_cache else None

        for i, block in enumerate(self.h):
            key_cache = key_caches[i] if key_caches is not None else None
            value_cache = value_caches[i] if value_caches is not None else None

            hidden_states, present = block(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                is_prefill=is_prefill,
                key_cache=key_cache,
                value_cache=value_cache,
                block_tables=block_tables,
                seq_lens=seq_lens,
                max_seq_len=max_seq_len,
            )

            if use_cache:
                presents.append(present)

        hidden_states = self.ln_f(hidden_states)

        return hidden_states, presents

class GPT2LMHeadModel(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        is_prefill: bool = True,
        key_caches: Optional[List[torch.Tensor]] = None,
        value_caches: Optional[List[torch.Tensor]] = None,
        block_tables: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        max_seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        hidden_states, presents = self.transformer(
            input_ids,
            position_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            is_prefill=is_prefill,
            key_caches=key_caches,
            value_caches=value_caches,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
        )

        lm_logits = self.lm_head(hidden_states)

        return lm_logits, presents