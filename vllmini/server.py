import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import GPT2Config, GPT2Tokenizer
import torch

from vllmini.scheduler import Scheduler
from vllmini.block_manager import BlockManager
from vllmini.model.gpt2 import GPT2LMHeadModel

class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 64

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
    
    num_blocks = 1000
    num_heads = config.num_attention_heads
    head_size = config.hidden_size // num_heads
    block_size = 16
    max_blocks_per_seq = 4
    
    block_manager = BlockManager(num_blocks, block_size, num_heads, head_size, max_blocks_per_seq)
    
    model = GPT2LMHeadModel(config)
    model.load_huggingface_weights("gpt2")
    model = model.to(device).to(torch.float16)
    
    # Initialize the scheduler with the model and block_manager
    scheduler = Scheduler(model, block_manager, max_length=20)  # You can adjust max_length as needed
    
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

    seq_id = scheduler.add_sequence(input_ids)

    # The sequence ID is now generated inside add_sequence

    return GenerationResponse(sequence_id=seq_id)

@app.get("/result/{seq_id}", response_model=ResultResponse)
async def get_result(seq_id: int):
    global scheduler, tokenizer
    
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler not initialized")

    if seq_id in scheduler.sequences:
        generated_ids = scheduler.sequences[seq_id]
        generated_tokens = tokenizer.decode(generated_ids[0].tolist())

        if seq_id in scheduler.active_sequences:
            return ResultResponse(status="in progress", generated=generated_tokens)

        scheduler.remove_completed_sequence(seq_id)
        return ResultResponse(status="completed", generated=generated_tokens)

    return ResultResponse(status="error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)