"""
Embedding Generation for Question-Answer Pairs
This script generates embeddings for Turkish QA pairs using the E5 model.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import Dict, List, Tuple
import time

# ===== CONFIGURATION SETTINGS =====
MODEL_IDENTIFIER = "ytu-ce-cosmos/turkish-e5-large"
INPUT_JSON_PATH = "qa_data_grouped.json"
OUTPUT_FILE_PATH = "qa_embeddings_minimized"
EMBEDDING_DEVICE = None  # Can be 'cuda' for GPU or None for auto-detection

class QAEmbeddingGenerator:
    """Handles embedding generation for question-answer pairs."""
    
    def __init__(self, model_name: str = MODEL_IDENTIFIER, device=None):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name/path of the sentence transformer model
            device: Computation device (None for auto-detection)
        """
        print(f"Loading model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name, device=device)
        self.embedding_size = self.embedding_model.get_sentence_embedding_dimension()
        print(f"Model embedding dimension: {self.embedding_size}")
    
    def format_for_e5(self, text_list: List[str]) -> List[str]:
        """
        Add E5-specific instruction prefix to texts.
        
        Args:
            text_list: List of raw text strings
            
        Returns:
            List of formatted texts with 'query: ' prefix
        """
        return [f"query: {text}" for text in text_list]
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            NumPy array of embeddings
        """
        formatted_texts = self.format_for_e5(texts)
        return self.embedding_model.encode(
            formatted_texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
    
    def prepare_dataset(self, qa_data: Dict[str, List[Dict]]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Process QA data and generate feature vectors.
        
        Args:
            qa_data: Dictionary with 'train' and 'test' keys containing QA pairs
            
        Returns:
            Dictionary with processed data for each split
        """
        processed_data = {}
        
        for split_name in ["train", "test"]:
            if split_name not in qa_data:
                print(f"Warning: '{split_name}' split not found in data")
                continue
                
            print(f"\n{'='*50}")
            print(f"Processing {split_name.upper()} data")
            print(f"{'='*50}")
            
            split_data = qa_data[split_name]
            
            # Extract components
            questions = [item['question'] for item in split_data]
            answers = [item['answer'] for item in split_data]
            labels = [item['label'] for item in split_data]
            
            # Generate embeddings
            print("Encoding questions...")
            question_vectors = self.create_embeddings(questions)
            
            print("Encoding answers...")
            answer_vectors = self.create_embeddings(answers)
            
            # Combine question and answer embeddings
            combined_features = self.combine_features(question_vectors, answer_vectors)
            
            # Prepare labels
            label_array = np.array(labels, dtype=np.float32).reshape(-1, 1)
            
            print(f"{split_name} features shape: {combined_features.shape}")
            print(f"{split_name} labels shape: {label_array.shape}")
            
            processed_data[split_name] = (combined_features, label_array)
            
        return processed_data
    
    def combine_features(self, q_vectors: np.ndarray, a_vectors: np.ndarray) -> np.ndarray:
        """
        Combine question and answer embeddings with bias term.
        
        Args:
            q_vectors: Question embeddings
            a_vectors: Answer embeddings
            
        Returns:
            Combined feature matrix with bias term
        """
        # Concatenate question and answer vectors
        combined = np.hstack([q_vectors, a_vectors])
        
        # Add bias column (all ones)
        bias_col = np.ones((combined.shape[0], 1), dtype=np.float32)
        
        return np.hstack([combined, bias_col])
    
    def save_results(self, processed_data: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                    output_path: str) -> None:
        """
        Save processed data to compressed numpy file.
        
        Args:
            processed_data: Dictionary with processed data
            output_path: Base path for output files
        """
        if "train" in processed_data and "test" in processed_data:
            X_train, Y_train = processed_data["train"]
            X_test, Y_test = processed_data["test"]
            
            output_file = f"{output_path}.npz"
            np.savez_compressed(
                output_file,
                X_train=X_train,
                Y_train=Y_train,
                X_test=X_test,
                Y_test=Y_test
            )
            
            print(f"\n✓ Data successfully saved to: {output_file}")
            
            # Verify saved data
            self.verify_saved_data(output_file)
        else:
            print("Error: Missing train or test data in processed results")
    
    def verify_saved_data(self, file_path: str) -> None:
        """
        Verify that saved data can be loaded correctly.
        
        Args:
            file_path: Path to saved .npz file
        """
        try:
            loaded_data = np.load(file_path)
            print("\nVerification of saved data:")
            print(f"  Training features shape: {loaded_data['X_train'].shape}")
            print(f"  Training labels shape: {loaded_data['Y_train'].shape}")
            print(f"  Test features shape: {loaded_data['X_test'].shape}")
            print(f"  Test labels shape: {loaded_data['Y_test'].shape}")
            
            # Check for any invalid values
            for key in ['X_train', 'Y_train', 'X_test', 'Y_test']:
                if np.any(np.isnan(loaded_data[key])):
                    print(f"  Warning: NaN values detected in {key}")
                if np.any(np.isinf(loaded_data[key])):
                    print(f"  Warning: Infinite values detected in {key}")
                    
            print("✓ Data verification complete")
            
        except Exception as e:
            print(f"✗ Error during verification: {e}")


def load_qa_data(file_path: str) -> Dict:
    """
    Load QA data from JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary containing QA data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        print(f"✓ Successfully loaded data from: {file_path}")
        return data
    except FileNotFoundError:
        print(f"✗ Error: File not found at {file_path}")
        raise
    except json.JSONDecodeError:
        print(f"✗ Error: Invalid JSON format in {file_path}")
        raise


def main():
    """Main execution function."""
    print("=" * 60)
    print("QA Embedding Generation Script")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Load QA data
        qa_dataset = load_qa_data(INPUT_JSON_PATH)
        
        # Initialize embedding generator
        generator = QAEmbeddingGenerator(
            model_name=MODEL_IDENTIFIER,
            device=EMBEDDING_DEVICE
        )
        
        # Process data
        results = generator.prepare_dataset(qa_dataset)
        
        # Save results
        generator.save_results(results, OUTPUT_FILE_PATH)
        
        elapsed_time = time.time() - start_time
        print(f"\n✓ Processing completed in {elapsed_time:.2f} seconds")
        
    except Exception as error:
        print(f"\n✗ An error occurred during execution:")
        print(f"  Error type: {type(error).__name__}")
        print(f"  Error message: {error}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)