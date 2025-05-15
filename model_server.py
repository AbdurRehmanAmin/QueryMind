#!/usr/bin/env python3
"""
Simple FastAPI server for serving QueryMindCoder model via API.
This allows the resource-intensive model to run separately from the main backend.
"""

import os
import argparse
import torch
import logging
from typing import Dict, List, Optional, Union, Any
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Model request and response schemas
class ModelRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=1000, gt=0, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stop: Optional[List[str]] = None
    stream: bool = False
    
class ModelResponse(BaseModel):
    generated_text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    processing_time: float

# Initialize FastAPI app
app = FastAPI(
    title="QueryMindCoder API",
    description="API for generating text with QueryMindCoder",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and tokenizer
model = None
tokenizer = None

def load_model(model_id: str = "AbdurRehmanAmin/QueryMindCoder"):
    """Load the model and tokenizer"""
    global model, tokenizer
    
    # Avoid reloading if already loaded
    if model is not None and tokenizer is not None:
        return
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Loading model: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        logger.info(f"Model loaded successfully: {model_id}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None or tokenizer is None:
        return {"status": "not_ready", "message": "Model not loaded yet"}
    return {"status": "ready", "model": "Qwen2.5-coder-7b"}

@app.post("/generate", response_model=ModelResponse)
async def generate_text(request: ModelRequest):
    """Generate text using the model"""
    global model, tokenizer
    
    # Ensure model is loaded
    if model is None or tokenizer is None:
        load_model()
    
    start_time = time.time()
    
    try:
        # Tokenize input
        inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
        
        # Get the number of input tokens
        input_tokens = len(inputs.input_ids[0])
        
        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                do_sample=request.temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
                stop_sequences=request.stop,
            )
        
        # Decode generated text
        generated_ids = outputs[0, inputs.input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Count tokens
        output_tokens = len(generated_ids)
        total_tokens = input_tokens + output_tokens
        
        # Apply stop sequences
        if request.stop:
            for stop_seq in request.stop:
                if stop_seq in generated_text:
                    generated_text = generated_text.split(stop_seq)[0]
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        logger.info(f"Generated {output_tokens} tokens in {processing_time:.2f}s")
        
        return ModelResponse(
            generated_text=generated_text,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=total_tokens,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")

@app.post("/")
async def compatibility_endpoint(request: Request):
    """OpenAI-compatible endpoint for easier integration"""
    try:
        body = await request.json()
        
        # Extract parameters from the request
        prompt = body.get("prompt", "")
        max_tokens = body.get("max_tokens", 1000)
        temperature = body.get("temperature", 0.7)
        stop = body.get("stop", None)
        
        # Create ModelRequest
        model_request = ModelRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )
        
        # Generate text
        result = await generate_text(model_request)
        
        # Format response like OpenAI API
        return {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "Qwen2.5-coder-7b",
            "choices": [
                {
                    "text": result.generated_text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop" if stop and any(s in result.generated_text for s in stop) else "length",
                }
            ],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.total_tokens,
            }
        }
        
    except Exception as e:
        logger.error(f"Error in compatibility endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

def parse_args():
    parser = argparse.ArgumentParser(description="QueryMindCoder API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server")
    parser.add_argument("--model-id", type=str, default="AbdurRehmanAmin/QueryMindCoder", help="Hugging Face model ID")
    parser.add_argument("--load-at-startup", action="store_true", help="Load model at startup")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Optionally load the model at startup
    if args.load_at_startup:
        load_model(args.model_id)
    
    # Start the server
    uvicorn.run("model_server:app", host=args.host, port=args.port, reload=False) 