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

        self.block_size = 16  # This can be made configurable
        self.max_blocks = 1024  # This can be made configurable

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
        q, k, v = qkv.split(self.hidden_size, dim=-1)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if is_prefill:
            attn_output = self._vanilla_attention(q, k, v, attention_mask)
        else:
            attn_output = self._paged_attention(q, key_cache, value_cache, block_tables, seq_lens, max_seq_len)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        attn_output = self.c_proj(attn_output)

        if use_cache:
            return attn_output, (k, v)
        return attn_output, None

    def _vanilla_attention(self, q, k, v, attention_mask):
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        return torch.matmul(attn_weights, v)

    def _paged_attention(self, q, key_cache, value_cache, block_tables, seq_lens, max_seq_len):
        batch_size, num_heads, seq_len, head_dim = q.shape
        out = torch.empty_like(q)

        # Reshape for paged attention kernel
        q = q.contiguous().view(batch_size * num_heads, seq_len, head_dim)

        paged_attention_v1(
            out.view(batch_size * num_heads, seq_len, head_dim),
            q,
            key_cache,
            value_cache,
            self.num_heads,
            self.scale,
            block_tables,
            seq_lens,
            self.block_size,
            max_seq_len,
            None,  # alibi_slopes
            "auto",
            1.0,  # kv_scale
            0,  # tp_rank
            0,  # blocksparse_local_blocks
            1,  # blocksparse_vert_stride
            1,  # blocksparse_block_size
            0,  # blocksparse_head_sliding_step
        )
        return out

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
            layer_block_tables = block_tables[i] if block_tables is not None else None

            hidden_states, present = block(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                is_prefill=is_prefill,
                key_cache=key_cache,
                value_cache=value_cache,
                block_tables=layer_block_tables,
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

    def load_huggingface_weights(self, model_name_or_path: str):
        from transformers import GPT2LMHeadModel as HFModel
        print(f"Loading weights from {model_name_or_path}")
        hf_model = HFModel.from_pretrained(model_name_or_path)
        hf_state_dict = hf_model.state_dict()

        # Create a mapping between HuggingFace state dict keys and our model's keys
        key_mapping = {
            'transformer.wte.weight': 'transformer.wte.weight',
            'transformer.wpe.weight': 'transformer.wpe.weight',
            'transformer.ln_f.weight': 'transformer.ln_f.weight',
            'transformer.ln_f.bias': 'transformer.ln_f.bias',
            'lm_head.weight': 'lm_head.weight',
        }

        # Add mappings for each layer
        for i in range(self.config.num_hidden_layers):
            hf_prefix = f'transformer.h.{i}.'
            our_prefix = f'transformer.h.{i}.'
            layer_mapping = {
                f'{hf_prefix}ln_1.weight': f'{our_prefix}ln_1.weight',
                f'{hf_prefix}ln_1.bias': f'{our_prefix}ln_1.bias',
                f'{hf_prefix}attn.c_attn.weight': f'{our_prefix}attn.c_attn.weight',
                f'{hf_prefix}attn.c_attn.bias': f'{our_prefix}attn.c_attn.bias',
                f'{hf_prefix}attn.c_proj.weight': f'{our_prefix}attn.c_proj.weight',
                f'{hf_prefix}attn.c_proj.bias': f'{our_prefix}attn.c_proj.bias',
                f'{hf_prefix}ln_2.weight': f'{our_prefix}ln_2.weight',
                f'{hf_prefix}ln_2.bias': f'{our_prefix}ln_2.bias',
                f'{hf_prefix}mlp.c_fc.weight': f'{our_prefix}mlp.c_fc.weight',
                f'{hf_prefix}mlp.c_fc.bias': f'{our_prefix}mlp.c_fc.bias',
                f'{hf_prefix}mlp.c_proj.weight': f'{our_prefix}mlp.c_proj.weight',
                f'{hf_prefix}mlp.c_proj.bias': f'{our_prefix}mlp.c_proj.bias',
            }
            key_mapping.update(layer_mapping)

        # Create a new state dict for our model
        new_state_dict = {}
        for hf_key, our_key in key_mapping.items():
            if hf_key in hf_state_dict:
                # Transpose weights for linear layers
                if 'attn.c_attn.weight' in hf_key or 'attn.c_proj.weight' in hf_key or 'mlp.c_fc.weight' in hf_key or 'mlp.c_proj.weight' in hf_key:
                    new_state_dict[our_key] = hf_state_dict[hf_key].t()
                else:
                    new_state_dict[our_key] = hf_state_dict[hf_key]
            else:
                print(f"Warning: Key {hf_key} not found in HuggingFace model")

        # Load the new state dict into our model
        missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
        
        if missing_keys:
            print(f"Warning: Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys: {unexpected_keys}")

        print("Weights loaded successfully")