"""
Embedding Generator Module for Vietnamese Cosmetics Recommender System

This module generates product embeddings using the AITeamVN/Vietnamese_Embedding model.
It takes bert_input_text from enriched products and produces dense vector embeddings.

Author: Senior Python Data Engineer
Date: 2025-11-22
"""

import logging
import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Configure logger
logger = logging.getLogger("data_layer")


class EmbeddingGenerator:
    """
    Generate product embeddings using Vietnamese BERT model.
    
    This class handles:
    - Loading the AITeamVN/Vietnamese_Embedding model
    - Batch processing for efficient GPU/CPU utilization
    - Mean pooling of token embeddings
    - Saving embeddings in PyTorch tensor format
    """
    
    def __init__(
        self, 
        model_name: str = "AITeamVN/Vietnamese_Embedding",
        batch_size: int = 32,
        max_length: int = 512,
        device: Optional[str] = None
    ):
        """
        Initialize the EmbeddingGenerator.
        
        Args:
            model_name: HuggingFace model identifier
            batch_size: Number of texts to process in each batch
            max_length: Maximum token length for BERT input
            device: Device to run model on ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Auto-detect device if not specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Initializing EmbeddingGenerator on device: {self.device}")
        
        # Load tokenizer and model
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        logger.info(f"Model loaded successfully. Embedding dimension: {self.model.config.hidden_size}")
    
    def mean_pooling(
        self, 
        token_embeddings: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply mean pooling to token embeddings to get sentence-level embeddings.
        
        This method averages token embeddings while respecting the attention mask
        (ignoring padding tokens).
        
        Args:
            token_embeddings: Token-level embeddings [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Sentence embeddings [batch_size, hidden_size]
        """
        # Expand attention mask to match embedding dimensions
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Sum embeddings (masked)
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        
        # Sum mask values (count of non-padding tokens)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        
        # Calculate mean
        return sum_embeddings / sum_mask
    
    def generate_embeddings(
        self, 
        texts: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts using batch processing.
        
        Args:
            texts: List of input texts to encode
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings [num_texts, hidden_size]
        """
        all_embeddings = []
        num_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        logger.info(f"Generating embeddings for {len(texts)} texts in {num_batches} batches")
        
        # Create progress bar iterator
        batch_iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            batch_iterator = tqdm(batch_iterator, desc="Generating embeddings", total=num_batches)
        
        with torch.no_grad():  # Disable gradient computation for inference
            for i in batch_iterator:
                # Get batch
                batch_texts = texts[i:i + self.batch_size]
                
                # Tokenize
                encoded_input = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                # Move to device
                encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
                
                # Generate embeddings
                model_output = self.model(**encoded_input)
                
                # Apply mean pooling
                embeddings = self.mean_pooling(
                    model_output.last_hidden_state,
                    encoded_input['attention_mask']
                )
                
                # Move to CPU and convert to numpy
                embeddings = embeddings.cpu().numpy()
                all_embeddings.append(embeddings)
        
        # Concatenate all batches
        all_embeddings = np.vstack(all_embeddings)
        
        logger.info(f"Generated embeddings shape: {all_embeddings.shape}")
        return all_embeddings
    
    def process_and_save(
        self,
        input_path: str,
        output_path: str,
        text_column: str = 'bert_input_text'
    ) -> Dict[str, any]:
        """
        Complete pipeline: load enriched data, generate embeddings, and save to PyTorch format.
        
        Args:
            input_path: Path to enriched products Parquet file
            output_path: Path where embeddings will be saved (.pt file)
            text_column: Column name containing text to encode
            
        Returns:
            Dictionary containing metadata about the generated embeddings
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If text_column is missing
        """
        try:
            # Load enriched data
            logger.info(f"Loading enriched data from: {input_path}")
            df = pd.read_parquet(input_path)
            logger.info(f"Loaded {len(df)} products")
            
            # Validate text column
            if text_column not in df.columns:
                raise ValueError(f"Column '{text_column}' not found in data. Available columns: {list(df.columns)}")
            
            # Extract texts
            texts = df[text_column].tolist()
            logger.info(f"Extracted {len(texts)} texts from column '{text_column}'")
            
            # Generate embeddings
            logger.info("Starting embedding generation...")
            embeddings = self.generate_embeddings(texts, show_progress=True)
            
            # Prepare output data structure
            output_data = {
                'embeddings': torch.from_numpy(embeddings),  # Convert to PyTorch tensor
                'product_ids': df['product_id'].tolist(),
                'metadata': {
                    'model_name': self.model_name,
                    'embedding_dim': embeddings.shape[1],
                    'num_products': embeddings.shape[0],
                    'max_length': self.max_length,
                    'text_column': text_column,
                    'device_used': self.device
                }
            }
            
            # Create output directory if it doesn't exist
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save to PyTorch format
            logger.info(f"Saving embeddings to: {output_path}")
            torch.save(output_data, output_path)
            
            logger.info(f"✓ Embedding generation complete. Output saved to {output_path}")
            logger.info(f"  - Total embeddings: {embeddings.shape[0]}")
            logger.info(f"  - Embedding dimension: {embeddings.shape[1]}")
            logger.info(f"  - File size: {Path(output_path).stat().st_size / (1024*1024):.2f} MB")
            
            return output_data['metadata']
            
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error during embedding generation: {e}")
            raise


def main():
    """
    Main execution block for testing the EmbeddingGenerator module.
    
    This demonstrates the complete pipeline with sample data paths.
    Adjust paths according to your project structure.
    """
    # Configure logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Define paths (adjust to your project structure)
    base_path = Path(__file__).parent.parent.parent.parent.parent
    input_path = base_path / "data" / "processed" / "enriched_products.parquet"
    output_path = base_path / "data" / "processed" / "content_based_embeddings" / "product_embeddings.pt"
    
    # Initialize generator
    generator = EmbeddingGenerator(
        model_name="AITeamVN/Vietnamese_Embedding",
        batch_size=32,  # Adjust based on your GPU memory
        max_length=512
    )
    
    # Run the pipeline
    try:
        metadata = generator.process_and_save(
            input_path=str(input_path),
            output_path=str(output_path),
            text_column='bert_input_text'
        )
        
        print("\n✓ Embedding generation pipeline executed successfully!")
        print(f"  Model: {metadata['model_name']}")
        print(f"  Products: {metadata['num_products']}")
        print(f"  Dimension: {metadata['embedding_dim']}")
        
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        logger.exception("Full traceback:")


if __name__ == "__main__":
    main()
