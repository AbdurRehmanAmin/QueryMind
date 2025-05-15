"""
Case Bank component for QueryMind.

This module implements the case bank and retrieval mechanisms for QueryMind's 
case-based reasoning system. It stores and retrieves relevant cases from 
past ML problem solutions.
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default cases directory - point to the actual data directory
DEFAULT_CASES_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

class Case:
    """Represents a single case in the case bank"""
    
    def __init__(self, 
                 task_name: str, 
                 task_description: str, 
                 solution_code: str,
                 embeddings: Optional[np.ndarray] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a case.
        
        Args:
            task_name: Unique identifier for the task
            task_description: Description of the ML problem
            solution_code: Python code that solves the problem
            embeddings: Vector representation of the case (optional)
            metadata: Additional information about the case (optional)
        """
        self.task_name = task_name
        self.task_description = task_description
        self.solution_code = solution_code
        self.embeddings = embeddings
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert case to dictionary for serialization"""
        result = {
            "task_name": self.task_name,
            "task_description": self.task_description,
            "solution_code": self.solution_code,
            "metadata": self.metadata
        }
        
        # Don't store embeddings in JSON
        if self.embeddings is not None:
            result["has_embeddings"] = True
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Case':
        """Create case from dictionary"""
        return cls(
            task_name=data["task_name"],
            task_description=data["task_description"],
            solution_code=data["solution_code"],
            metadata=data.get("metadata", {})
        )


class CaseBank:
    """
    Manages a collection of cases and provides retrieval mechanisms.
    """
    
    def __init__(self, cases_dir: str = DEFAULT_CASES_DIR):
        """
        Initialize the case bank.
        
        Args:
            cases_dir: Directory containing the case files
        """
        self.cases_dir = cases_dir
        self.cases: List[Case] = []
        self.embeddings_model = None
        
        # Create cases directory if it doesn't exist
        os.makedirs(self.cases_dir, exist_ok=True)
        
        # Load existing cases
        self._load_cases()
    
    def _load_cases(self):
        """Load all cases from the cases directory and its subdirectories"""
        # Check if the cases directory exists
        if not os.path.exists(self.cases_dir):
            logger.warning(f"Cases directory {self.cases_dir} does not exist")
            return
            
        # Get all subdirectories (tabular_cases, nlp_cases, tsa_cases)
        subdirs = [d for d in os.listdir(self.cases_dir) 
                  if os.path.isdir(os.path.join(self.cases_dir, d))]
        
        # If no subdirectories, check for case files in the main directory
        if not subdirs:
            self._load_cases_from_dir(self.cases_dir)
        else:
            # Load cases from each subdirectory
            for subdir in subdirs:
                subdir_path = os.path.join(self.cases_dir, subdir)
                # Add category to metadata
                self._load_cases_from_dir(subdir_path, {'category': subdir})
                
        logger.info(f"Loaded {len(self.cases)} cases in total")
    
    def _load_cases_from_dir(self, directory: str, extra_metadata: Dict[str, Any] = None):
        """Load cases from a specific directory"""
        case_files = [f for f in os.listdir(directory) 
                     if f.endswith('.case')]
        
        for file_name in case_files:
            try:
                file_path = os.path.join(directory, file_name)
                # Read the case file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    case_content = f.read()
                
                # Parse file name to get task name (remove .case extension)
                task_name = os.path.splitext(file_name)[0]
                
                # Create a case with the file content as the task description
                # and empty solution code for now (will be filled later)
                metadata = {'source_file': file_path}
                if extra_metadata:
                    metadata.update(extra_metadata)
                    
                case = Case(
                    task_name=task_name,
                    task_description=case_content[:1000],  # First 1000 chars as description
                    solution_code=case_content,  # Full content as solution code
                    metadata=metadata
                )
                self.cases.append(case)
                
            except Exception as e:
                logger.error(f"Error loading case {file_name}: {e}")
        
        logger.info(f"Loaded {len(self.cases)} cases from {directory}")
    
    def add_case(self, case: Case):
        """
        Add a new case to the case bank.
        
        Args:
            case: The case to add
        """
        # Add to in-memory collection
        self.cases.append(case)
        
        # Determine appropriate directory based on metadata
        category = case.metadata.get('category', 'general')
        target_dir = os.path.join(self.cases_dir, category)
        os.makedirs(target_dir, exist_ok=True)
        
        # Save to disk
        file_path = os.path.join(target_dir, f"{case.task_name}.case")
        
        # For now, just save the solution_code as the full content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(case.solution_code)
    
    def _ensure_embeddings(self):
        """Ensure all cases have embeddings"""
        if not self.embeddings_model:
            try:
                # Try to load the model
                import torch
                from transformers import AutoModel, AutoTokenizer
                
                logger.info("Loading embedding model")
                model_name = "sentence-transformers/all-mpnet-base-v2"
                self.embeddings_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.embeddings_model = AutoModel.from_pretrained(model_name)
                
                # Move to GPU if available
                if torch.cuda.is_available():
                    self.embeddings_model = self.embeddings_model.to("cuda")
            except Exception as e:
                logger.error(f"Error loading embedding model: {e}")
                return False
        
        # Generate embeddings for cases that don't have them
        for case in self.cases:
            if case.embeddings is None:
                try:
                    case.embeddings = self._generate_embeddings(case.task_description)
                except Exception as e:
                    logger.error(f"Error generating embeddings for {case.task_name}: {e}")
        
        return True
    
    def _generate_embeddings(self, text: str) -> np.ndarray:
        """Generate embeddings for text using the model"""
        import torch
        
        # Tokenize and prepare for model
        inputs = self.embeddings_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Move to same device as model
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.embeddings_model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        # Return the embedding vector
        return embeddings[0]
    
    def retrieve_similar_cases(self, query: str, top_k: int = 3) -> List[Tuple[Case, float]]:
        """
        Retrieve the most similar cases to the query.
        
        Args:
            query: The problem description to match against
            top_k: Number of cases to return
            
        Returns:
            List of (case, similarity_score) tuples
        """
        # Ensure all cases have embeddings
        if not self._ensure_embeddings():
            # Fall back to keyword-based retrieval if embeddings fail
            return self._keyword_retrieval(query, top_k)
        
        try:
            # Generate embeddings for the query
            query_embedding = self._generate_embeddings(query)
            
            # Calculate similarities
            similarities = []
            for case in self.cases:
                if case.embeddings is not None:
                    # Cosine similarity
                    similarity = np.dot(query_embedding, case.embeddings) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(case.embeddings)
                    )
                    similarities.append((case, float(similarity)))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error in embedding-based retrieval: {e}")
            # Fall back to keyword retrieval
            return self._keyword_retrieval(query, top_k)
    
    def _keyword_retrieval(self, query: str, top_k: int = 3) -> List[Tuple[Case, float]]:
        """Simple keyword-based retrieval as fallback"""
        # Tokenize query into words
        query_words = set(query.lower().split())
        
        # Calculate word overlap
        similarities = []
        for case in self.cases:
            case_words = set(case.task_description.lower().split())
            # Jaccard similarity
            intersection = len(query_words.intersection(case_words))
            union = len(query_words.union(case_words))
            similarity = intersection / union if union > 0 else 0
            similarities.append((case, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def generate_case_prompt(self, query: str, top_k: int = 3) -> str:
        """
        Generate a prompt with relevant case examples for the query.
        
        Args:
            query: The problem description
            top_k: Number of cases to include
            
        Returns:
            Formatted prompt with similar cases
        """
        similar_cases = self.retrieve_similar_cases(query, top_k)
        
        if not similar_cases:
            return "No similar cases found. Please solve the task from scratch."
        
        prompt = "Here are some similar cases that might help you solve this problem:\n\n"
        
        for i, (case, similarity) in enumerate(similar_cases, 1):
            prompt += f"CASE {i}:\n"
            prompt += f"PROBLEM: {case.task_description}\n"
            prompt += f"SOLUTION:\n```python\n{case.solution_code}\n```\n\n"
        
        prompt += "Please use these cases as references to solve the current problem."
        
        return prompt


# Create a default case bank instance
default_case_bank = CaseBank()

def add_solved_problem_to_case_bank(task_name: str, task_description: str, solution_code: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Add a successfully solved problem to the case bank.
    
    Args:
        task_name: Unique identifier for the task
        task_description: Description of the ML problem
        solution_code: Python code that solves the problem
        metadata: Additional information about the case (optional)
    """
    case = Case(
        task_name=task_name,
        task_description=task_description,
        solution_code=solution_code,
        metadata=metadata
    )
    default_case_bank.add_case(case)
    return case

def retrieve_similar_cases(query: str, top_k: int = 3) -> List[Tuple[Case, float]]:
    """
    Retrieve the most similar cases to the query from the default case bank.
    
    Args:
        query: The problem description to match against
        top_k: Number of cases to return
            
    Returns:
        List of (case, similarity_score) tuples
    """
    return default_case_bank.retrieve_similar_cases(query, top_k)

def generate_case_prompt(query: str, top_k: int = 3) -> str:
    """
    Generate a prompt with relevant case examples for the query.
    
    Args:
        query: The problem description
        top_k: Number of cases to include
            
    Returns:
        Formatted prompt with similar cases
    """
    return default_case_bank.generate_case_prompt(query, top_k) 