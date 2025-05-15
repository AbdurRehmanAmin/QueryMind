"""QueryMindCoder agent implementation with multi-agent architecture."""
import os
import sys
import json
import time
import torch
import subprocess
from Agents&Initializations.LLM import get_model_completion
from Agents&Initializations.schema import Action

class HFDSAgent:
    """
    Data Science Agent using Hugging Face models with multi-agent architecture:
    - Retriever: Retrieves similar cases
    - RankReviser: Evaluates and refines retrieved cases
    - Planner: Creates experiment plans
    - Programmer: Implements the plans
    - Debugger: Fixes code issues
    - Logger: Documents progress
    """

    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.research_problem = env.research_problem if hasattr(env, "research_problem") else ""
        self.log_dir = getattr(args, "log_dir", "logs")
        self.work_dir = getattr(env, "work_dir", "workspace")
        
        # Model settings
        self.model_id = getattr(args, "hf_model_id", None)
        self.api_url = getattr(args, "api_url", os.environ.get("MODEL_API_URL"))
        self.api_key = getattr(args, "api_key", os.environ.get("MODEL_API_KEY"))
        self.device = getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # Training settings
        self.max_training_time = getattr(args, "max_training_time", 3600)  # 1 hour default
        self.gpu_memory_limit = getattr(args, "gpu_memory_limit", None)
        
        # Agent roles and their respective model configurations
        self.agents = {
            "retriever": {"model_id": self.model_id, "api_url": self.api_url, "api_key": self.api_key},
            "rankreviser": {"model_id": self.model_id, "api_url": self.api_url, "api_key": self.api_key},
            "planner": {"model_id": self.model_id, "api_url": self.api_url, "api_key": self.api_key},
            "programmer": {"model_id": self.model_id, "api_url": self.api_url, "api_key": self.api_key},
            "debugger": {"model_id": self.model_id, "api_url": self.api_url, "api_key": self.api_key},
            "logger": {"model_id": self.model_id, "api_url": self.api_url, "api_key": self.api_key},
        }
        
        if self.api_url:
            print(f"Initialized QueryMind with remote model API: {self.api_url}")
        else:
            print(f"Initialized QueryMind with QueryMindCoder: {self.model_id} on {self.device}")
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize result dictionary to store trained model and metrics
        self.result = {
            "trained_model_path": None,
            "metrics": {},
            "execution_time": 0,
            "log_path": self.log_dir
        }
    
    def run(self, env):
        start_time = time.time()
        step = 0
        experiment_step = 0
        
        running_log = f"""
[Initial State] Launching data science project with Hugging Face model to solve: {self.research_problem}
"""
        with open(os.path.join(self.log_dir, "main_log"), "a", 1) as f:
            f.write(f"Step {step}" + ":\n")
            f.write(running_log + "\n")
        
        max_iterations = getattr(self.args, "max_iterations", 5)
        
        while not self.is_final(env) and step < max_iterations * 2:
            # 1. Develop the experiment plan using case-based reasoning
            action = "Develop Experiment Plan"
            action_input = {
                "experiment_log": running_log
            }
            plans = self.execute_action(env, action, action_input)
            step += 1
                
            with open(os.path.join(self.log_dir, "main_log"), "a", 1) as f:
                f.write(f"Step {step}" + ":\n")
                f.write(f"Action: {action}" + "\nObservation:\n" + plans + "\n") 
                
            # 2. Execute the experiment plan
            action = "Execute Experiment Plan"
            action_input = {
                "script_name": "train.py",
                "plan": plans,
                "save_name": "train.py"
            }
            execution_log = self.execute_action(env, action, action_input)
            step += 1
            experiment_step += 1
            
            with open(os.path.join(self.log_dir, "main_log"), "a", 1) as f:
                f.write(f"Step {step}" + ":\n")
                f.write(f"Action: {action}" + "\nObservation:\n" + execution_log + "\n")
            
            # 3. Log experiment results
            log_content = self.log_experiment(running_log, plans, execution_log, log_file=os.path.join(self.log_dir, "tmp.txt"))
            running_log += f"\n{log_content}"
            
            with open(os.path.join(self.log_dir, "main_log"), "a", 1) as f:
                f.write(f"Step {step}" + ":\n")
                f.write(running_log + "\n")

        # After all iterations, train the final model
        self.train_final_model()
        
        end_time = time.time()
        self.result["execution_time"] = end_time - start_time
        
        if self.is_final(env):
            return self.result
        else:
            return self.result

    def is_final(self, env):
        """Check if the environment is in a final state"""
        if hasattr(env, "is_final"):
            return env.is_final()
        return False

    def execute_action(self, env, action_name, action_input):
        """Execute an action in the environment"""
        if hasattr(env, "execute"):
            try:
                result = env.execute(Action(action_name, action_input))
                return result
            except Exception as e:
                print(f"Error executing action {action_name}: {e}")
                return f"Error: {str(e)}"
        else:
            # If no environment is provided, simulate with model
            prompt = f"""
Research Problem: {self.research_problem}

Action: {action_name}
Input: {json.dumps(action_input, indent=2)}

Execute the above action and provide detailed results.
"""
            agent_role = "planner" if "Plan" in action_name else "programmer"
            agent_config = self.agents[agent_role]
            
            return get_model_completion(
                prompt=prompt,
                model_id=agent_config.get("model_id"),
                api_url=agent_config.get("api_url"),
                api_key=agent_config.get("api_key"),
                max_tokens=1000,
                temperature=0.7,
                log_file=os.path.join(self.log_dir, f"{agent_role}_action.log")
            )

    def log_experiment(self, running_log, instructions, execution_log, log_file=None):
        """Log experiment results using the logger agent"""
        prompt = f"""Given the instructions and execution log of the last experiment on the research problem: 
        {instructions} 
        [Execution Log]:
        ```
        {execution_log}
        ```
        Here is the running log of your experiment:
        [Running Log]:
        ```
        {running_log}
        ```
        Summarize and append the progress of the last step to the running log in this format:
        [Experiment Summary]: According to the instructions, summarize what was experimented in the last step objectively.
        [Experiment Result]: According to the execution log and the running log, summarize if the last step of experiment brings performance improvement objectively. Only report the performance if this is the first experiment result.
        Do not include any result that is guessed rather than directly confirmed by the observation.
        """

        logger_config = self.agents["logger"]
        log = "[Experiment Summary]:" + get_model_completion(
            prompt=prompt,
            model_id=logger_config.get("model_id"),
            api_url=logger_config.get("api_url"),
            api_key=logger_config.get("api_key"),
            max_tokens=500,
            temperature=0.3,
            log_file=log_file
        ).split("[Experiment Summary]:")[1]
        
        return log
        
    def train_final_model(self):
        """Train the final model with the generated train.py script"""
        train_script = os.path.join(self.work_dir, "train.py")
        model_dir = os.path.join(self.work_dir, "model")
        os.makedirs(model_dir, exist_ok=True)
        
        # Add output directives to train.py to save the model and metrics
        self._update_train_script(train_script, model_dir)
        
        print(f"Training final model with {train_script}...")
        
        # Set environment variables for GPU memory management if needed
        env_vars = os.environ.copy()
        if self.gpu_memory_limit:
            env_vars["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:{self.gpu_memory_limit}"
        
        # Run the training script with a timeout
        try:
            result = subprocess.run(
                ["python", train_script],
                cwd=self.work_dir,
                capture_output=True,
                text=True,
                timeout=self.max_training_time,
                env=env_vars
            )
            
            # Save the output
            with open(os.path.join(self.log_dir, "training_output.log"), "w") as f:
                f.write(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")
                
            # Extract metrics from output
            self._extract_metrics(result.stdout)
            
            # Set trained model path if exists
            if os.path.exists(model_dir):
                self.result["trained_model_path"] = model_dir
                
            print(f"Model training completed. Output saved to {self.log_dir}/training_output.log")
            
        except subprocess.TimeoutExpired:
            print(f"Model training exceeded time limit of {self.max_training_time} seconds and was terminated")
            with open(os.path.join(self.log_dir, "training_output.log"), "w") as f:
                f.write("TIMEOUT: Training exceeded maximum allowed time and was terminated")
        except Exception as e:
            print(f"Error during model training: {e}")
            with open(os.path.join(self.log_dir, "training_output.log"), "w") as f:
                f.write(f"ERROR: {str(e)}")
    
    def _update_train_script(self, script_path, model_dir):
        """Add code to save model and metrics to the train.py script"""
        try:
            with open(script_path, "r") as f:
                script_content = f.read()
                
            # Check if the script already has model saving code
            if "save(" not in script_content and "save_pretrained(" not in script_content:
                # Add model saving code at the end
                save_code = f"""
# Save the trained model and metrics
import os
import json

# Create model directory
os.makedirs('{model_dir}', exist_ok=True)

# Save model artifacts
try:
    if 'model' in locals():
        if hasattr(model, 'save_pretrained'):
            model.save_pretrained('{model_dir}')
        elif hasattr(model, 'save'):
            torch.save(model.state_dict(), os.path.join('{model_dir}', 'model.pt'))
        else:
            import pickle
            with open(os.path.join('{model_dir}', 'model.pkl'), 'wb') as f:
                pickle.dump(model, f)
                
    # Save metrics
    metrics = {{}}
    for var_name in ['accuracy', 'precision', 'recall', 'f1', 'mse', 'mae', 'rmse', 'r2']:
        if var_name in locals():
            metrics[var_name] = float(locals()[var_name])
            
    with open(os.path.join('{model_dir}', 'metrics.json'), 'w') as f:
        json.dump(metrics, f)
        
    print("MODEL_SAVED: Model and metrics saved to {model_dir}")
except Exception as e:
    print(f"Error saving model: {{e}}")
"""
                with open(script_path, "a") as f:
                    f.write(save_code)
                    
        except Exception as e:
            print(f"Error updating train script: {e}")
    
    def _extract_metrics(self, output):
        """Extract metrics from the training output"""
        metrics = {}
        
        # Look for common metrics in the output
        metric_patterns = {
            "accuracy": r"(?:accuracy|acc)[:\s=]+([0-9.]+)",
            "precision": r"precision[:\s=]+([0-9.]+)",
            "recall": r"recall[:\s=]+([0-9.]+)",
            "f1": r"f1[:\s=]+([0-9.]+)",
            "mse": r"(?:mse|mean squared error)[:\s=]+([0-9.]+)",
            "mae": r"(?:mae|mean absolute error)[:\s=]+([0-9.]+)",
            "rmse": r"(?:rmse|root mean squared error)[:\s=]+([0-9.]+)",
            "r2": r"(?:r2|r\^2|r squared)[:\s=]+([0-9.]+)",
        }
        
        import re
        for metric_name, pattern in metric_patterns.items():
            matches = re.findall(pattern, output, re.IGNORECASE)
            if matches:
                try:
                    metrics[metric_name] = float(matches[-1])  # Use the last match
                except ValueError:
                    pass
                    
        # Check if metrics.json was created
        metrics_file = os.path.join(self.work_dir, "model", "metrics.json")
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, "r") as f:
                    file_metrics = json.load(f)
                # Update with file metrics (they take precedence)
                metrics.update(file_metrics)
            except:
                pass
                
        self.result["metrics"] = metrics 
