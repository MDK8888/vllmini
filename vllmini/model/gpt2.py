import torch
from torch import nn
from transformers import GPT2Config
from typing import List, Optional, Tuple
from paged_attention_cuda import paged_attention_v1, cache_ops

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
        slot_mapping: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        max_seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.c_attn(hidden_states)
        q, k, v = qkv.split(self.hidden_size, dim=-1)

        # Reshape q, k, v to [batch_size * seq_len, num_heads, head_dim]
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_heads, self.head_dim)
        v = v.view(-1, self.num_heads, self.head_dim)

        # Always cache k and v using the reshape_and_cache kernel
        self._cache_kv(k, v, key_cache, value_cache, slot_mapping)

        if is_prefill:
            # For prefill, we need to use the attention mask and perform full attention
            # Reshape q, k, v back to [batch_size, seq_len, num_heads, head_dim]
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
            k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
            v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
            
            # Transpose to [batch_size, num_heads, seq_len, head_dim]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            attn_output = self._vanilla_attention(q, k, v, attention_mask)
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        else:
            # For decoding, use paged attention
            attn_output = self._paged_attention(q, key_cache, value_cache, block_table, seq_lens, max_seq_len)
            attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

        attn_output = self.c_proj(attn_output)

        if use_cache:
            return attn_output, (key_cache, value_cache)
        return attn_output, None

    def _vanilla_attention(self, q, k, v, attention_mask):
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        return torch.matmul(attn_weights, v)

    def _cache_kv(self, k: torch.Tensor, v: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor, slot_mapping: torch.Tensor):
        cache_ops.reshape_and_cache(
            k,
            v,
            key_cache,
            value_cache,
            slot_mapping,
            "auto",  # kv_cache_dtype
            1.0,  # kv_scale
        )

    def _paged_attention(self, q, key_cache, value_cache, block_table, seq_lens, max_seq_len):
        num_seqs, num_heads, head_dim = q.shape
        out = torch.empty_like(q)
        print("query:", q)
        print("block_table:", block_table)
        print("seq_lens:", seq_lens)
        print("block_size:", self.block_size)
        print("max_seq_len:", max_seq_len)
        paged_attention_v1(
            out,  # [num_seqs, num_heads, head_dim]
            q,    # [num_seqs, num_heads, head_dim]
            key_cache,
            value_cache,
            self.num_heads,
            self.scale,
            block_table,
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
        slot_mapping: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        max_seq_len: Optional[int] = None,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            is_prefill=is_prefill,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=slot_mapping,
            block_table=block_table,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
        )
        attn_output = attn_outputs[0]
        assert attn_output.shape == residual.shape, f"Your attention output doesn't match the residual. attn_output shape: {attn_output.shape}, residual shape: {residual.shape}"

        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_output = self.mlp(hidden_states)
        hidden_states = feed_forward_output + residual

        outputs = hidden_states
        if use_cache:
            outputs = (outputs, attn_outputs[1])

        return outputs

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
        key_cache: Optional[torch.Tensor] = None,
        value_cache: Optional[torch.Tensor] = None,
        slot_mappings: Optional[List[torch.Tensor]] = None,
        block_tables: Optional[List[torch.Tensor]] = None,
        seq_lens: Optional[torch.Tensor] = None,
        max_seq_len: Optional[int] = None,
    ):
        batch_size, seq_length = input_ids.size()
        hidden_states = self.wte(input_ids) + self.wpe(position_ids)
        presents = () if use_cache else None

        for i, block in enumerate(self.h):
            slot_mapping = slot_mappings[i] if slot_mappings is not None else None
            block_table = block_tables[i] if block_tables is not None else None

            outputs, kv_cache = block(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                is_prefill=is_prefill,
                key_cache=key_cache,
                value_cache=value_cache,
                slot_mapping=slot_mapping,
                block_table=block_table,
                seq_lens=seq_lens,
                max_seq_len=max_seq_len,
            )

            key_cache, value_cache = kv_cache[0], kv_cache[1]

            hidden_states = outputs[0].unsqueeze(0)
            if use_cache:
                presents = (key_cache, value_cache)

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
        key_cache: Optional[torch.Tensor] = None,
        value_cache: Optional[torch.Tensor] = None,
        slot_mappings: Optional[List[torch.Tensor]] = None,
        block_tables: Optional[List[torch.Tensor]] = None,
        seq_lens: Optional[torch.Tensor] = None,
        max_seq_len: Optional[int] = None,
    ):
        transformer_outputs = self.transformer(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            is_prefill=is_prefill,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mappings=slot_mappings,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        return lm_logits, transformer_outputs[1] if use_cache else None

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