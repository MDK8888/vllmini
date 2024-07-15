import asyncio
from fastapi import FastAPI, BackgroundTasks
from contextlib import asynccontextmanager
from pydantic import BaseModel
from transformers import GPT2Config, GPT2Tokenizer
import torch

from .kv_cache import KVCache
from .paged_attention import PagedAttention
from .scheduler import Scheduler

class GenerationRequest(BaseModel):
    prompt: str
    max_length: int

class GenerationResponse(BaseModel):
    sequence_id: int

sequence_counter = 0
paged_attention = None
scheduler = None
tokenizer = None
sequences = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global paged_attention, scheduler, tokenizer
    
    # Initialize components
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = GPT2Config.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    num_layers = config.n_layer
    num_blocks = 1000
    num_heads = config.num_attention_heads
    head_size = config.hidden_size // num_heads
    block_size = 128
    kv_cache = KVCache(num_layers, num_blocks, num_heads, head_size, block_size)
    paged_attention = PagedAttention("gpt2", kv_cache, device)
    
    # Initialize the scheduler with the paged_attention object
    scheduler = Scheduler(paged_attention)
    
    # Start the scheduler
    scheduler_task = asyncio.create_task(scheduler_runner())
    
    yield
    
    # Shutdown
    scheduler_task.cancel()
    try:
        await scheduler_task
    except asyncio.CancelledError:
        pass

async def scheduler_runner():
    while True:
        scheduler.run()
        await asyncio.sleep(0.01)  # Small delay to prevent CPU hogging

app = FastAPI(lifespan=lifespan)

@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    global sequence_counter
    sequence_counter += 1
    seq_id = sequence_counter

    tokens = tokenizer.encode(request.prompt)
    input_ids = torch.tensor([tokens], dtype=torch.int64, device=paged_attention.device)
    position_ids = torch.arange(len(tokens), dtype=torch.int64, device=paged_attention.device).unsqueeze(0)
    attention_mask = torch.ones((1, 1, len(tokens), len(tokens)), dtype=torch.float32, device=paged_attention.device)
    slot_mapping = torch.arange(len(tokens), dtype=torch.int64, device=paged_attention.device)

    scheduler.add_sequence(seq_id, input_ids, position_ids, attention_mask, slot_mapping, request.max_length)

    return GenerationResponse(sequence_id=seq_id)

@app.get("/result/{seq_id}")
async def get_result(seq_id: int):
    status = scheduler.get_sequence_status(seq_id)
    if status is None:
        return {"status": "not found"}
    if not status["completed"]:
        return {"status": "in progress", "generated": tokenizer.decode(status["tokens"])}
    return {"status": "completed", "generated": tokenizer.decode(status["tokens"])}