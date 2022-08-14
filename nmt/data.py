"""
Data loading and preprocessing pipeline.

This module handles loading the OPUS medical corpus and preprocessing it
for neural machine translation training. It includes tokenization, bucketing,
and masking operations.
"""

import trax
from .utils import append_eos, VOCAB_FILE, VOCAB_DIR


def load_data():
    """
    Load the English-German translation dataset from OPUS.
    
    Uses the OPUS medical corpus for this project. It's a smaller dataset
    that's perfect for demonstrating the architecture without requiring extensive
    computational resources.
    
    Returns:
        tuple: (train_stream_fn, eval_stream_fn) - Generator functions for training and evaluation data
    """
    # Get generator function for the training set
    train_stream_fn = trax.data.TFDS('opus/medical',
                                     data_dir='./data/',
                                     keys=('en', 'de'),
                                     eval_holdout_size=0.01,
                                     train=True)

    # Get generator function for the eval set
    eval_stream_fn = trax.data.TFDS('opus/medical',
                                    data_dir='./data/',
                                    keys=('en', 'de'),
                                    eval_holdout_size=0.01,
                                    train=False)
    
    return train_stream_fn, eval_stream_fn


def prepare_data_pipeline(train_stream_fn, eval_stream_fn, 
                          vocab_file=None, vocab_dir=None,
                          max_length=512):
    """
    Build the complete data preprocessing pipeline.
    
    This pipeline handles tokenization, filtering, bucketing, and masking.
    Bucketing is particularly important - it groups sentences of similar length
    together to minimize padding and improve training efficiency.
    
    Args:
        train_stream_fn: Training data generator function
        eval_stream_fn: Evaluation data generator function
        vocab_file (str): Vocabulary filename (default: from utils)
        vocab_dir (str): Vocabulary directory path (default: from utils)
        max_length (int): Maximum sequence length (default: 512)
        
    Returns:
        tuple: (train_batch_stream, eval_batch_stream) ready for training
    """
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
    boundaries = [8, 16, 32, 64, 128, 256, 512]
    batch_sizes = [256, 128, 64, 32, 16, 8, 4, 2]
    
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
    train_batch_stream = trax.data.AddLossWeights(id_to_mask=0)(train_batch_stream)
    eval_batch_stream = trax.data.AddLossWeights(id_to_mask=0)(eval_batch_stream)
    
    return train_batch_stream, eval_batch_stream


def get_data_pipeline(max_length=512):
    """
    Convenience function to get complete data pipeline in one call.
    
    Args:
        max_length (int): Maximum sequence length (default: 512)
        
    Returns:
        tuple: (train_batch_stream, eval_batch_stream)
    """
    train_stream_fn, eval_stream_fn = load_data()
    return prepare_data_pipeline(train_stream_fn, eval_stream_fn, max_length=max_length)

