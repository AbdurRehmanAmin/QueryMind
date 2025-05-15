""" Module for calling Hugging Face models either locally or via remote APIs. """

import os
import json
import torch
import requests
import tiktoken
from typing import List, Dict, Any, Optional
from .schema import LLMError

# Use cl100k_base tokenizer for logging
try:
    enc = tiktoken.get_encoding("cl100k_base")
except:
    print("Warning: tiktoken not installed or cl100k_base not available")

def log_to_file(log_file, prompt, completion, model, max_tokens_to_sample):
    """ Log the prompt and completion to a file."""
    if not log_file:
        return
        
    with open(log_file, "a") as f:
        f.write("\n===================prompt=====================\n")
        f.write(prompt)
        try:
            num_prompt_tokens = len(enc.encode(prompt))
            num_sample_tokens = len(enc.encode(completion))
            f.write(f"\n==================={model} response ({max_tokens_to_sample})=====================\n")
            f.write(completion)
            f.write("\n===================tokens=====================\n")
            f.write(f"Number of prompt tokens: {num_prompt_tokens}\n")
            f.write(f"Number of sampled tokens: {num_sample_tokens}\n")
        except:
            f.write(f"\n==================={model} response ({max_tokens_to_sample})=====================\n")
            f.write(completion)
        f.write("\n\n")

def complete_text_huggingface(prompt, model_id, stop_sequences=[], max_tokens_to_sample=1000, temperature=0.5, log_file=None, **kwargs):
    """ Call a local Hugging Face model to complete a prompt."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        generate_ids = model.generate(
            inputs.input_ids, 
            max_new_tokens=max_tokens_to_sample,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            **kwargs
        )
        
        # Get only the newly generated tokens
        generated_ids = generate_ids[0, inputs.input_ids.shape[1]:]
        completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Apply stop sequences if provided
        if stop_sequences:
            for stop_seq in stop_sequences:
                if stop_seq in completion:
                    completion = completion.split(stop_seq)[0]
        
        if log_file is not None:
            log_to_file(log_file, prompt, completion, model_id, max_tokens_to_sample)
        
        return completion
    except Exception as e:
        print(f"Error using Hugging Face model: {e}")
        raise LLMError(e)

def complete_text_remote_api(prompt, api_url, api_key=None, stop_sequences=[], max_tokens_to_sample=1000, temperature=0.5, log_file=None, **kwargs):
    """Call a remote API endpoint (like a hosted Qwen2.5-coder-7b) to complete a prompt."""
    try:
        headers = {
            "Content-Type": "application/json"
        }
        
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens_to_sample,
            "temperature": temperature,
            "stop": stop_sequences if stop_sequences else None,
            **kwargs
        }
        
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        
        if response.status_code != 200:
            raise LLMError(f"API call failed with status code {response.status_code}: {response.text}")
            
        result = response.json()
        
        # Handle different API response formats
        if "choices" in result and len(result["choices"]) > 0:
            # OpenAI-like format
            if "message" in result["choices"][0]:
                completion = result["choices"][0]["message"]["content"]
            else:
                completion = result["choices"][0]["text"]
        elif "generated_text" in result:
            # HF Inference Endpoints format
            completion = result["generated_text"]
        elif "completion" in result:
            # Custom format
            completion = result["completion"]
        else:
            raise LLMError(f"Unknown API response format: {result}")
            
        # Apply stop sequences if needed (some APIs don't handle them)
        if stop_sequences:
            for stop_seq in stop_sequences:
                if stop_seq in completion:
                    completion = completion.split(stop_seq)[0]
        
        if log_file is not None:
            log_to_file(log_file, prompt, completion, api_url, max_tokens_to_sample)
            
        return completion
    except Exception as e:
        print(f"Error calling remote API: {e}")
        raise LLMError(e)

def get_model_completion(prompt, model_id=None, api_url=None, api_key=None, stop_sequences=[], max_tokens=1000, temperature=0.5, log_file=None, **kwargs):
    """
    Get completions from either a local Hugging Face model or a remote API.
    
    Args:
        prompt: The text prompt to complete
        model_id: The Hugging Face model ID (for local models)
        api_url: URL to a remote model API (if using remote inference)
        api_key: API key for remote API (if required)
        stop_sequences: List of sequences that stop generation
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        log_file: Path to log file
        kwargs: Additional arguments for the model or API
        
    Returns:
        Completed text from the model
    """
    # Check environment variables for API settings if not provided
    if api_url is None:
        api_url = os.environ.get("MODEL_API_URL", None)
    
    if api_key is None:
        api_key = os.environ.get("MODEL_API_KEY", None)
        
    # Default model ID if nothing specified
    if model_id is None and api_url is None:
        model_id = os.environ.get("MODEL_ID", "gpt2")
    
    # Use remote API if URL is provided
    if api_url:
        return complete_text_remote_api(
            prompt, 
            api_url, 
            api_key, 
            stop_sequences, 
            max_tokens, 
            temperature, 
            log_file, 
            **kwargs
        )
    # Otherwise use local Hugging Face model
    else:
        return complete_text_huggingface(
            prompt, 
            model_id, 
            stop_sequences, 
            max_tokens, 
            temperature, 
            log_file, 
            **kwargs
        )

