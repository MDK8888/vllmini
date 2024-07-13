import asyncio
from fastapi import FastAPI, BackgroundTasks
from contextlib import asynccontextmanager
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from .kv_cache import KVCache
from .paged_attention import PagedAttention
from .scheduler import Scheduler
from .sequence_manager import SequenceManager

class GenerationRequest(BaseModel):
    prompt: str
    max_length: int

class GenerationResponse(BaseModel):
    sequence_id: int

sequence_counter = 0
paged_attention = None
seq_manager = None
scheduler = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global paged_attention, seq_manager, scheduler
    
    # Initialize components
    model = GPT2LMHeadModel.from_pretrained("gpt2").to('cuda')
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    kv_cache = KVCache(num_blocks=1000, num_heads=12, head_size=64, block_size=128)
    paged_attention = PagedAttention(model, kv_cache)
    seq_manager = SequenceManager()
    scheduler = Scheduler(paged_attention, seq_manager)
    
    # Start the scheduler
    scheduler_task = asyncio.create_task(scheduler.run())
    
    yield
    
    # Shutdown
    scheduler_task.cancel()
    try:
        await scheduler_task
    except asyncio.CancelledError:
        pass

app = FastAPI(lifespan=lifespan)

@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest, background_tasks: BackgroundTasks):
    global sequence_counter
    sequence_counter += 1
    seq_id = sequence_counter

    tokens = paged_attention.model.tokenizer.encode(request.prompt)
    seq_manager.add_sequence(seq_id, tokens, request.max_length)
    background_tasks.add_task(scheduler.add_sequence, seq_id)

    return GenerationResponse(sequence_id=seq_id)

@app.get("/result/{seq_id}")
async def get_result(seq_id: int):
    seq = seq_manager.sequences.get(seq_id)
    if seq is None:
        return {"status": "not found"}
    if seq_id in seq_manager.active_sequences:
        return {"status": "in progress", "generated": paged_attention.model.tokenizer.decode(seq.tokens)}
    return {"status": "completed", "generated": paged_attention.model.tokenizer.decode(seq.tokens)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)