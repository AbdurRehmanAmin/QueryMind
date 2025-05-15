#!/usr/bin/env python3
"""
Run a Hugging Face model-based DS-Agent on a custom dataset.
For use as a backend service for a web frontend.
"""

import os
import argparse
import shutil
import json
import torch
import tempfile
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from development.MLAgentBench.agents.hf_agent import HFDSAgent

class SimpleEnvironment:
    """A simple environment for running the HF DS-Agent"""
    
    def __init__(self, args):
        self.args = args
        self.research_problem = args.task_description
        self.work_dir = args.work_dir
        
        # Create work directory if it doesn't exist
        os.makedirs(self.work_dir, exist_ok=True)
        
        # Set up environment
        self._setup_environment()
    
    def _setup_environment(self):
        """Set up the environment with dataset and train.py scaffold"""
        # Create train.py scaffold if it doesn't exist
        train_py_path = os.path.join(self.work_dir, "train.py")
        if not os.path.exists(train_py_path):
            with open(train_py_path, "w") as f:
                f.write("""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error

# Load and prepare data
# TODO: Implement data loading and preprocessing

# Create model
# TODO: Implement model creation and training

# Train model
# TODO: Implement model training

# Evaluate model
# TODO: Implement model evaluation

# Print results
print("Accuracy: 0.0")  # Placeholder
""")
        
        # Copy dataset to target directory if it exists
        if os.path.exists(self.args.dataset):
            if os.path.isdir(self.args.dataset):
                # Copy all files in the directory
                for file in os.listdir(self.args.dataset):
                    src_file = os.path.join(self.args.dataset, file)
                    dst_file = os.path.join(self.work_dir, file)
                    if os.path.isfile(src_file):
                        shutil.copy2(src_file, dst_file)
                        print(f"Copied {src_file} to {dst_file}")
            else:
                # Copy the single file
                dst_file = os.path.join(self.work_dir, os.path.basename(self.args.dataset))
                shutil.copy2(self.args.dataset, dst_file)
                print(f"Copied {self.args.dataset} to {dst_file}")
                
        # Write task description to file
        with open(os.path.join(self.work_dir, "research_problem.txt"), "w") as f:
            f.write(self.research_problem)
    
    def is_final(self):
        """Check if the environment is in a final state"""
        return False  # Always let the agent hit max iterations

    def execute(self, action):
        """Execute an action in the environment"""
        action_name = action.name
        action_input = action.input_dict
        
        if action_name == "Develop Experiment Plan":
            # Use a Hugging Face model to generate a plan
            from transformers import pipeline
            
            # If API URL is provided, use it instead of local model
            if self.args.api_url:
                import requests
                
                headers = {
                    "Content-Type": "application/json"
                }
                
                if self.args.api_key:
                    headers["Authorization"] = f"Bearer {self.args.api_key}"
                    
                prompt = f"""
Research Problem: {self.research_problem}

Current state: {action_input.get('experiment_log', '')}

Create a detailed plan to solve this problem.
Step-by-step approach:
1. 
"""
                
                payload = {
                    "prompt": prompt,
                    "max_tokens": 500,
                    "temperature": 0.7,
                }
                
                try:
                    response = requests.post(self.args.api_url, headers=headers, json=payload, timeout=60)
                    
                    if response.status_code != 200:
                        return f"API Error: {response.status_code} - {response.text}"
                        
                    result = response.json()
                    
                    # Handle different API response formats
                    if "choices" in result and len(result["choices"]) > 0:
                        if "message" in result["choices"][0]:
                            text = result["choices"][0]["message"]["content"]
                        else:
                            text = result["choices"][0]["text"]
                    elif "generated_text" in result:
                        text = result["generated_text"]
                    else:
                        text = str(result)
                        
                    if "Create a detailed plan to solve this problem" in text:
                        plan = text.split("Create a detailed plan to solve this problem.")[1]
                    else:
                        plan = text
                        
                    return plan
                    
                except Exception as e:
                    return f"Error calling API: {str(e)}"
            
            else:
                # Use local model
                try:
                    pipe = pipeline(
                        "text-generation", 
                        model=self.args.hf_model_id,
                        device=0 if torch.cuda.is_available() else -1
                    )
                    
                    prompt = f"""
Research Problem: {self.research_problem}

Current state: {action_input.get('experiment_log', '')}

Create a detailed plan to solve this problem.
Step-by-step approach:
1. 
"""
                    result = pipe(prompt, max_new_tokens=500, temperature=0.7)[0]['generated_text']
                    plan = result.split("Create a detailed plan to solve this problem.")[1]
                    return plan
                    
                except Exception as e:
                    return f"Error generating plan: {str(e)}"
            
        elif action_name == "Execute Experiment Plan":
            # Implement the plan in train.py
            script_name = action_input.get("script_name", "train.py")
            plan = action_input.get("plan", "")
            save_name = action_input.get("save_name", "train.py")
            
            try:
                # Generate improved train.py based on the plan
                if self.args.api_url:
                    # Use remote API
                    import requests
                    
                    headers = {
                        "Content-Type": "application/json"
                    }
                    
                    if self.args.api_key:
                        headers["Authorization"] = f"Bearer {self.args.api_key}"
                    
                    # Read existing script
                    with open(os.path.join(self.work_dir, script_name), "r") as f:
                        existing_code = f.read()
                    
                    prompt = f"""
Research Problem: {self.research_problem}

Implementation Plan:
{plan}

Existing Code:
```python
{existing_code}
```

Write an improved implementation that follows the plan. Include all necessary imports, data loading, model definition, training, and evaluation.

```python
"""
                    
                    payload = {
                        "prompt": prompt,
                        "max_tokens": 800,
                        "temperature": 0.5,
                    }
                    
                    try:
                        response = requests.post(self.args.api_url, headers=headers, json=payload, timeout=120)
                        
                        if response.status_code != 200:
                            return f"API Error: {response.status_code} - {response.text}"
                            
                        result = response.json()
                        
                        # Handle different API response formats
                        if "choices" in result and len(result["choices"]) > 0:
                            if "message" in result["choices"][0]:
                                text = result["choices"][0]["message"]["content"]
                            else:
                                text = result["choices"][0]["text"]
                        elif "generated_text" in result:
                            text = result["generated_text"]
                        else:
                            text = str(result)
                        
                        # Extract code from response
                        if "```python" in text:
                            parts = text.split("```python")
                            if len(parts) > 1:
                                code_parts = parts[1].split("```")
                                if len(code_parts) > 0:
                                    new_code = code_parts[0].strip()
                                else:
                                    new_code = parts[1].strip()
                            else:
                                new_code = text
                        else:
                            new_code = text
                            
                    except Exception as e:
                        return f"Error calling API: {str(e)}"
                
                else:
                    # Use local model
                    from transformers import pipeline
                    
                    pipe = pipeline(
                        "text-generation", 
                        model=self.args.hf_model_id,
                        device=0 if torch.cuda.is_available() else -1
                    )
                    
                    # Read existing script
                    with open(os.path.join(self.work_dir, script_name), "r") as f:
                        existing_code = f.read()
                    
                    prompt = f"""
Research Problem: {self.research_problem}

Implementation Plan:
{plan}

Existing Code:
```python
{existing_code}
```

Write an improved implementation that follows the plan. Include all necessary imports, data loading, model definition, training, and evaluation.

```python
"""
                    result = pipe(prompt, max_new_tokens=800, temperature=0.5)[0]['generated_text']
                    if "```python" in result:
                        parts = result.split("```python")
                        if len(parts) > 1:
                            code_parts = parts[1].split("```")
                            if len(code_parts) > 0:
                                new_code = code_parts[0].strip()
                            else:
                                new_code = parts[1].strip()
                        else:
                            new_code = result
                    else:
                        new_code = result.split("```")[0].strip()
                
                # Save the improved implementation
                with open(os.path.join(self.work_dir, save_name), "w") as f:
                    f.write(new_code)
                
                # Test running the script (just syntax checking)
                import subprocess
                try:
                    result = subprocess.run(
                        ["python", "-m", "py_compile", save_name],
                        cwd=self.work_dir,
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    if result.returncode != 0:
                        return f"Code syntax check failed:\n{result.stderr}"
                    
                    return f"Successfully generated and syntax-checked implementation based on the plan."
                    
                except Exception as e:
                    return f"Code generation succeeded but syntax check failed: {str(e)}"
                
            except Exception as e:
                return f"Error executing plan: {str(e)}"
        
        return f"Unknown action: {action_name}"

def parse_args():
    parser = argparse.ArgumentParser(description="Run a Hugging Face model-based DS-Agent on a custom dataset")
    
    # Dataset and environment settings
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset file or directory")
    parser.add_argument("--task_description", type=str, required=True, help="Description of the task to solve")
    parser.add_argument("--work_dir", type=str, default="workspace", help="Working directory for the agent")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory for logs")
    
    # Model settings
    parser.add_argument("--hf_model_id", type=str, default="gpt2", help="Hugging Face model ID for local execution")
    parser.add_argument("--api_url", type=str, default=None, help="URL to remote model API")
    parser.add_argument("--api_key", type=str, default=None, help="API key for remote model API")
    parser.add_argument("--max_iterations", type=int, default=5, help="Maximum number of iterations")
    
    # Training settings
    parser.add_argument("--max_training_time", type=int, default=3600, help="Maximum training time in seconds")
    parser.add_argument("--gpu_memory_limit", type=int, default=None, help="GPU memory limit in MB")
    
    # System settings
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use for model inference")
    
    # Output settings
    parser.add_argument("--output_file", type=str, default=None, help="Path to output JSON file")
    parser.add_argument("--include_model", action="store_true", help="Include model files in output")
    
    args = parser.parse_args()
    return args

def encode_model_files(model_dir):
    """Encode model files as base64 strings"""
    if not os.path.exists(model_dir):
        return None
        
    model_files = {}
    for root, _, files in os.walk(model_dir):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, model_dir)
            try:
                with open(file_path, "rb") as f:
                    content = f.read()
                    model_files[rel_path] = base64.b64encode(content).decode('utf-8')
            except:
                print(f"Failed to encode {file_path}")
                
    return model_files

def run_agent(args):
    """Run the DS-Agent and return results as a dictionary"""
    # Prepare directories
    work_dir = os.path.abspath(args.work_dir)
    log_dir = os.path.abspath(args.log_dir)
    args.work_dir = work_dir
    args.log_dir = log_dir
    
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up environment
    print(f"Setting up environment in {work_dir}")
    env = SimpleEnvironment(args)
    
    # Initialize agent
    if args.api_url:
        print(f"Initializing HF DS-Agent with remote API: {args.api_url}")
    else:
        print(f"Initializing HF DS-Agent with model: {args.hf_model_id}")
    
    agent = HFDSAgent(args, env)
    
    # Run agent
    print("Running agent...")
    result = agent.run(env)
    print(f"Agent finished: {result}")
    
    # Include model files if requested
    if args.include_model and result.get("trained_model_path"):
        model_files = encode_model_files(result["trained_model_path"])
        result["model_files"] = model_files
    
    # Clean result for JSON output
    clean_result = {
        "metrics": result.get("metrics", {}),
        "execution_time": result.get("execution_time", 0),
        "log_path": result.get("log_path", ""),
        "success": True
    }
    
    if args.include_model:
        clean_result["model_files"] = result.get("model_files", {})
    
    # Save output to file if specified
    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(clean_result, f)
    
    return clean_result

def main():
    args = parse_args()
    result = run_agent(args)
    
    # Print final result
    print(f"Execution Time: {result['execution_time']:.2f} seconds")
    print(f"Metrics: {json.dumps(result['metrics'], indent=2)}")
    print(f"Logs: {result['log_path']}")
    
    # Print model files info if included
    if "model_files" in result:
        print(f"Model Files: {len(result['model_files'])} files included")
    
    # Print final model location
    print(f"Trained model and results are available in: {args.work_dir}")

def api_handler(event, context=None):
    """
    AWS Lambda or similar cloud function handler
    Expects event with the following structure:
    {
        "dataset_url": "https://example.com/dataset.csv",
        "task_description": "Predict housing prices based on features",
        "model_api_url": "https://api.example.com/model",
        "model_api_key": "your-api-key",
        "max_iterations": 3
    }
    """
    try:
        # Download dataset from URL
        import requests
        from urllib.parse import urlparse
        
        dataset_url = event.get("dataset_url")
        task_description = event.get("task_description")
        
        if not dataset_url or not task_description:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing required parameters: dataset_url and task_description"})
            }
        
        # Create temp directory for this run
        work_dir = tempfile.mkdtemp()
        log_dir = os.path.join(work_dir, "logs")
        
        # Download dataset
        response = requests.get(dataset_url)
        if response.status_code != 200:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": f"Failed to download dataset: {response.status_code}"})
            }
        
        # Save dataset
        parsed_url = urlparse(dataset_url)
        filename = os.path.basename(parsed_url.path)
        if not filename:
            filename = "dataset.csv"
            
        dataset_path = os.path.join(work_dir, filename)
        with open(dataset_path, "wb") as f:
            f.write(response.content)
        
        # Set up args
        class Args:
            pass
            
        args = Args()
        args.dataset = dataset_path
        args.task_description = task_description
        args.work_dir = work_dir
        args.log_dir = log_dir
        args.hf_model_id = event.get("hf_model_id", "gpt2")
        args.api_url = event.get("model_api_url")
        args.api_key = event.get("model_api_key")
        args.max_iterations = event.get("max_iterations", 3)
        args.max_training_time = event.get("max_training_time", 1800)  # 30 min default
        args.gpu_memory_limit = event.get("gpu_memory_limit")
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        args.output_file = None
        args.include_model = event.get("include_model", True)
        
        # Run agent
        result = run_agent(args)
        
        # Clean up temp files unless specified to keep them
        if not event.get("keep_files", False):
            shutil.rmtree(work_dir)
        
        return {
            "statusCode": 200,
            "body": json.dumps(result)
        }
        
    except Exception as e:
        import traceback
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e),
                "traceback": traceback.format_exc()
            })
        }

if __name__ == "__main__":
    main() 