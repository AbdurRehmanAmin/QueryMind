#!/usr/bin/env python3
"""
FastAPI backend for connecting web frontend to our QueryMind system.
This API serves as the bridge between the frontend and the QueryMind implementation.
"""

import os
import shutil
import tempfile
import logging
import uuid
import json
import base64
from typing import Dict, List, Optional, Any, Union
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import asyncio
from urllib.parse import unquote

# Import our QueryMind code
from run_hf_agent import run_agent, Args

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Define the models for request and response
class DSAgentRequest(BaseModel):
    task_description: str
    model_api_url: Optional[str] = None
    model_api_key: Optional[str] = None
    max_iterations: int = 3
    max_training_time: int = 1800  # 30 minutes
    include_model: bool = True

class DSAgentResponse(BaseModel):
    task_id: str
    status: str
    message: str

class TaskResult(BaseModel):
    task_id: str
    status: str
    metrics: Optional[Dict[str, float]] = None
    execution_time: Optional[float] = None
    model_files: Optional[Dict[str, str]] = None
    error: Optional[str] = None

# Initialize FastAPI app
app = FastAPI(
    title="QueryMind Backend API",
    description="API for running QueryMind on machine learning tasks",
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

# In-memory storage for task status
tasks = {}

# Create a directory for storing uploads
UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Create a directory for storing results
RESULTS_DIR = os.path.join(os.getcwd(), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

@app.get("/")
async def root():
    """Root endpoint for checking if the server is running"""
    return {"message": "QueryMind Backend API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/run-dsagent", response_model=DSAgentResponse)
async def run_dsagent(
    background_tasks: BackgroundTasks,
    task_description: str = Form(...),
    dataset: UploadFile = File(...),
    model_api_url: Optional[str] = Form(None),
    model_api_key: Optional[str] = Form(None),
    max_iterations: int = Form(3),
    max_training_time: int = Form(1800),
    include_model: bool = Form(True)
):
    """Run QueryMind with the uploaded dataset"""
    # Generate a unique task ID
    task_id = str(uuid.uuid4())
    
    # Create a directory for this task
    task_dir = os.path.join(UPLOAD_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)
    
    # Create results directory for this task
    result_dir = os.path.join(RESULTS_DIR, task_id)
    os.makedirs(result_dir, exist_ok=True)
    
    # Save the uploaded file
    dataset_path = os.path.join(task_dir, dataset.filename)
    with open(dataset_path, "wb") as f:
        f.write(await dataset.read())
    
    # Initialize task status
    tasks[task_id] = {
        "status": "queued",
        "message": "Task queued",
        "task_description": task_description,
        "dataset": dataset_path,
        "model_api_url": model_api_url,
        "model_api_key": model_api_key,
        "max_iterations": max_iterations,
        "max_training_time": max_training_time,
        "include_model": include_model,
        "result": None,
        "error": None
    }
    
    # Run the task in the background
    background_tasks.add_task(
        process_task, 
        task_id, 
        task_description, 
        dataset_path, 
        model_api_url, 
        model_api_key, 
        max_iterations, 
        max_training_time, 
        include_model,
        result_dir
    )
    
    return DSAgentResponse(
        task_id=task_id,
        status="queued",
        message="Task submitted successfully"
    )

@app.get("/task/{task_id}", response_model=TaskResult)
async def get_task_status(task_id: str):
    """Get the status of a task"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    
    result = TaskResult(
        task_id=task_id,
        status=task["status"],
    )
    
    # Add result if available
    if task["result"] is not None:
        result.metrics = task["result"].get("metrics", {})
        result.execution_time = task["result"].get("execution_time", 0)
        result.model_files = task["result"].get("model_files", {})
    
    # Add error if available
    if task["error"] is not None:
        result.error = task["error"]
    
    return result

async def process_task(
    task_id: str,
    task_description: str,
    dataset_path: str,
    model_api_url: Optional[str],
    model_api_key: Optional[str],
    max_iterations: int,
    max_training_time: int,
    include_model: bool,
    result_dir: str
):
    """Process a QueryMind task"""
    try:
        # Update task status
        tasks[task_id]["status"] = "running"
        tasks[task_id]["message"] = "Running QueryMind"
        
        # Set up args
        args = Args()
        args.dataset = dataset_path
        args.task_description = task_description
        args.work_dir = os.path.join(result_dir, "workspace")
        args.log_dir = os.path.join(result_dir, "logs")
        args.hf_model_id = "gpt2"  # Default, will be overridden by API if provided
        args.api_url = model_api_url
        args.api_key = model_api_key
        args.max_iterations = max_iterations
        args.max_training_time = max_training_time
        args.device = "cuda" if os.environ.get("USE_CUDA", "False").lower() == "true" else "cpu"
        args.output_file = os.path.join(result_dir, "result.json")
        args.include_model = include_model
        
        # Run the agent
        logger.info(f"Running QueryMind for task {task_id}")
        result = run_agent(args)
        
        # Update task status
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["message"] = "Task completed successfully"
        tasks[task_id]["result"] = result
        
        # Save result
        with open(os.path.join(result_dir, "result.json"), "w") as f:
            json.dump(result, f, indent=2)
            
        logger.info(f"Task {task_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Error processing task {task_id}: {str(e)}")
        import traceback
        error_details = traceback.format_exc()
        
        # Update task status
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["message"] = f"Task failed: {str(e)}"
        tasks[task_id]["error"] = error_details
        
        # Save error
        with open(os.path.join(result_dir, "error.log"), "w") as f:
            f.write(error_details)

@app.get("/download/{task_id}")
async def download_model(task_id: str):
    """Download the model files for a completed task"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not completed")
    
    if task["result"] is None:
        raise HTTPException(status_code=400, detail="No result available")
    
    if "model_files" not in task["result"] or not task["result"]["model_files"]:
        raise HTTPException(status_code=400, detail="No model files available")
    
    # Return the model files
    return JSONResponse(content={"model_files": task["result"]["model_files"]})

@app.delete("/task/{task_id}")
async def delete_task(task_id: str):
    """Delete a task and its files"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Delete task directories
    task_dir = os.path.join(UPLOAD_DIR, task_id)
    result_dir = os.path.join(RESULTS_DIR, task_id)
    
    if os.path.exists(task_dir):
        shutil.rmtree(task_dir)
    
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    
    # Remove task from memory
    del tasks[task_id]
    
    return {"message": f"Task {task_id} deleted successfully"}

def parse_args():
    import argparse
    
    parser = argparse.ArgumentParser(description="QueryMind Backend API")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind the server")
    parser.add_argument("--use-cuda", action="store_true", help="Use CUDA for inference if available")
    parser.add_argument("--model-api-url", type=str, help="Default model API URL")
    parser.add_argument("--model-api-key", type=str, help="Default model API key")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Set environment variables
    os.environ["USE_CUDA"] = "True" if args.use_cuda else "False"
    if args.model_api_url:
        os.environ["MODEL_API_URL"] = args.model_api_url
    if args.model_api_key:
        os.environ["MODEL_API_KEY"] = args.model_api_key
    
    # Start the server
    uvicorn.run("backend_api:app", host=args.host, port=args.port, reload=False) 