import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import GPT2Config, GPT2Tokenizer
import torch

from .scheduler import Scheduler
from .paged_attention import PagedAttention
from .kv_cache import KVCache

class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 100

class GenerationResponse(BaseModel):
    sequence_id: int

class ResultResponse(BaseModel):
    status: str
    generated: str = None

# Global variables
scheduler = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global scheduler, tokenizer, device
    
    # Initialize components
    config = GPT2Config.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    num_layers = 12
    num_blocks = 1000
    num_heads = config.num_attention_heads
    head_size = config.hidden_size // num_heads
    block_size = 128
    kv_cache = KVCache(num_layers, num_blocks, num_heads, head_size, block_size)
    paged_attention = PagedAttention("gpt2", kv_cache, device)
    
    # Initialize the scheduler with the paged_attention object
    scheduler = Scheduler(paged_attention, max_length=1024)  # You can adjust max_length as needed
    
    # Start the scheduler in a background task
    scheduler_task = asyncio.create_task(run_scheduler())
    
    yield
    
    # Shutdown
    scheduler_task.cancel()
    try:
        await scheduler_task
    except asyncio.CancelledError:
        pass

async def run_scheduler():
    while True:
        scheduler.run()
        await asyncio.sleep(0.01)  # Small delay to prevent CPU hogging

app = FastAPI(lifespan=lifespan)

@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    global scheduler, tokenizer, device
    
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler not initialized")

    tokens = tokenizer.encode(request.prompt)
    input_ids = torch.tensor([tokens], dtype=torch.int64, device=device)
    position_ids = torch.arange(len(tokens), dtype=torch.int64, device=device).unsqueeze(0)
    attention_mask = torch.ones((1, len(tokens)), dtype=torch.float32, device=device)

    seq_id = len(scheduler.active_sequences) + 1  # Simple way to generate unique seq_id
    scheduler.add_sequence(seq_id, input_ids, position_ids, attention_mask)

    return GenerationResponse(sequence_id=seq_id)

@app.get("/result/{seq_id}", response_model=ResultResponse)
async def get_result(seq_id: int):
    global scheduler, tokenizer
    
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler not initialized")

    if seq_id not in scheduler.active_sequences and seq_id not in scheduler.block_tables:
        return ResultResponse(status="not found")
    
    if seq_id in scheduler.active_sequences:
        return ResultResponse(status="in progress")
    
    if seq_id in scheduler.block_tables:
        generated_tokens = []
        for block in scheduler.block_tables[seq_id]:
            block_tokens = scheduler.kv_cache.get_block_tokens(block)
            generated_tokens.extend(block_tokens)
        
        generated_text = tokenizer.decode(generated_tokens)
        return ResultResponse(status="completed", generated=generated_text)

    return ResultResponse(status="error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)