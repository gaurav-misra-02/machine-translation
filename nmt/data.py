"""
Data loading and preprocessing pipeline.

This module handles loading the OPUS medical corpus and preprocessing it
for neural machine translation training. It includes tokenization, bucketing,
and masking operations.
"""

from typing import Callable, Generator, Optional, Tuple
import trax
from .utils import append_eos, VOCAB_FILE, VOCAB_DIR

# Constants for data pipeline configuration
DEFAULT_MAX_LENGTH = 512
DEFAULT_BOUNDARIES = [8, 16, 32, 64, 128, 256, 512]
DEFAULT_BATCH_SIZES = [256, 128, 64, 32, 16, 8, 4, 2]
EVAL_HOLDOUT_SIZE = 0.01
PADDING_ID = 0


def load_data(data_dir: str = './data/') -> Tuple[Callable, Callable]:
    """
    Load the English-German translation dataset from OPUS.
    
    Uses the OPUS medical corpus for this project. It's a smaller dataset
    that's perfect for demonstrating the architecture without requiring extensive
    computational resources.
    
    Args:
        data_dir: Directory to store/load the data
    
    Returns:
        (train_stream_fn, eval_stream_fn) - Generator functions for training and evaluation data
    
    Raises:
        RuntimeError: If the dataset cannot be loaded
    """
    try:
        # Get generator function for the training set
        train_stream_fn = trax.data.TFDS('opus/medical',
                                         data_dir=data_dir,
                                         keys=('en', 'de'),
                                         eval_holdout_size=EVAL_HOLDOUT_SIZE,
                                         train=True)

        # Get generator function for the eval set
        eval_stream_fn = trax.data.TFDS('opus/medical',
                                        data_dir=data_dir,
                                        keys=('en', 'de'),
                                        eval_holdout_size=EVAL_HOLDOUT_SIZE,
                                        train=False)
        
        return train_stream_fn, eval_stream_fn
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {str(e)}") from e


def prepare_data_pipeline(train_stream_fn: Callable, 
                          eval_stream_fn: Callable, 
                          vocab_file: Optional[str] = None, 
                          vocab_dir: Optional[str] = None,
                          max_length: int = DEFAULT_MAX_LENGTH) -> Tuple[Generator, Generator]:
    """
    Build the complete data preprocessing pipeline.
    
    This pipeline handles tokenization, filtering, bucketing, and masking.
    Bucketing is particularly important - it groups sentences of similar length
    together to minimize padding and improve training efficiency.
    
    Args:
        train_stream_fn: Training data generator function
        eval_stream_fn: Evaluation data generator function
        vocab_file: Vocabulary filename (default: from utils)
        vocab_dir: Vocabulary directory path (default: from utils)
        max_length: Maximum sequence length (must be positive)
        
    Returns:
        (train_batch_stream, eval_batch_stream) ready for training
    
    Raises:
        ValueError: If max_length is not positive
    """
    if max_length <= 0:
        raise ValueError(f"max_length must be positive, got {max_length}")
    if vocab_file is None:
        vocab_file = VOCAB_FILE
    if vocab_dir is None:
        vocab_dir = VOCAB_DIR
    
    # Tokenize the datasets
    tokenized_train_stream = trax.data.Tokenize(
        vocab_file=vocab_file, vocab_dir=vocab_dir
    )(train_stream_fn())
    
    tokenized_eval_stream = trax.data.Tokenize(
        vocab_file=vocab_file, vocab_dir=vocab_dir
    )(eval_stream_fn())
    
    # Append EOS tokens
    tokenized_train_stream = append_eos(tokenized_train_stream)
    tokenized_eval_stream = append_eos(tokenized_eval_stream)
    
    # Filter long sentences
    filtered_train_stream = trax.data.FilterByLength(
        max_length=max_length, length_keys=[0, 1]
    )(tokenized_train_stream)
    
    filtered_eval_stream = trax.data.FilterByLength(
        max_length=max_length, length_keys=[0, 1]
    )(tokenized_eval_stream)
    
    # Bucketing configuration - sentences grouped by similar length
    # Short sentences get larger batch sizes, long sentences get smaller batches
    boundaries = DEFAULT_BOUNDARIES
    batch_sizes = DEFAULT_BATCH_SIZES
    
    # Create bucketed batches
    train_batch_stream = trax.data.BucketByLength(
        boundaries, batch_sizes,
        length_keys=[0, 1]
    )(filtered_train_stream)
    
    eval_batch_stream = trax.data.BucketByLength(
        boundaries, batch_sizes,
        length_keys=[0, 1]
    )(filtered_eval_stream)
    
    # Add masking for padding
    train_batch_stream = trax.data.AddLossWeights(id_to_mask=PADDING_ID)(train_batch_stream)
    eval_batch_stream = trax.data.AddLossWeights(id_to_mask=PADDING_ID)(eval_batch_stream)
    
    return train_batch_stream, eval_batch_stream


def get_data_pipeline(max_length: int = DEFAULT_MAX_LENGTH) -> Tuple[Generator, Generator]:
    """
    Convenience function to get complete data pipeline in one call.
    
    Args:
        max_length: Maximum sequence length
        
    Returns:
        (train_batch_stream, eval_batch_stream)
    
    Raises:
        ValueError: If max_length is not positive
        RuntimeError: If dataset cannot be loaded
    """
    train_stream_fn, eval_stream_fn = load_data()
    return prepare_data_pipeline(train_stream_fn, eval_stream_fn, max_length=max_length)

