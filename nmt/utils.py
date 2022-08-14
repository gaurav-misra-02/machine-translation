"""
Utility functions for tokenization and text processing.

This module provides helper functions for converting between text and tokens,
which are essential for preprocessing inputs and postprocessing model outputs.
"""

import numpy as np
import trax


# Global vocabulary configuration
VOCAB_FILE = 'ende_32k.subword'
VOCAB_DIR = 'data/'
EOS = 1  # End-of-sentence token


def tokenize(input_str, vocab_file=None, vocab_dir=None):
    """
    Encode a string to an array of integer tokens.
    
    Uses subword tokenization to handle out-of-vocabulary words gracefully.
    This approach breaks words into subword units, allowing the model to handle
    rare or unseen words.

    Args:
        input_str (str): Human-readable string to encode
        vocab_file (str): Filename of the vocabulary text file
        vocab_dir (str): Path to the vocabulary file
  
    Returns:
        numpy.ndarray: Tokenized version of the input string with shape (1, n_tokens)
    """
    if vocab_file is None:
        vocab_file = VOCAB_FILE
    if vocab_dir is None:
        vocab_dir = VOCAB_DIR
        
    # Use the trax.data.tokenize method
    inputs = next(trax.data.tokenize(iter([input_str]),
                                     vocab_file=vocab_file, vocab_dir=vocab_dir))
    
    # Mark the end of the sentence with EOS
    inputs = list(inputs) + [EOS]
    
    # Add the batch dimension
    batch_inputs = np.reshape(np.array(inputs), [1, -1])
    
    return batch_inputs


def detokenize(integers, vocab_file=None, vocab_dir=None):
    """
    Decode an array of integers to a human readable string.

    Args:
        integers (numpy.ndarray or list): Array of integers to decode
        vocab_file (str): Filename of the vocabulary text file
        vocab_dir (str): Path to the vocabulary file
  
    Returns:
        str: The decoded sentence
    """
    if vocab_file is None:
        vocab_file = VOCAB_FILE
    if vocab_dir is None:
        vocab_dir = VOCAB_DIR
        
    # Remove dimensions of size 1
    integers = list(np.squeeze(integers))
    
    # Remove the EOS to decode only the original tokens
    if EOS in integers:
        integers = integers[:integers.index(EOS)] 
    
    return trax.data.detokenize(integers, vocab_file=vocab_file, vocab_dir=vocab_dir)


def append_eos(stream):
    """
    Generator that appends EOS token to each sentence in the stream.
    
    Args:
        stream: Generator yielding (input, target) tuples
        
    Yields:
        tuple: (input with EOS, target with EOS)
    """
    for (inputs, targets) in stream:
        inputs_with_eos = list(inputs) + [EOS]
        targets_with_eos = list(targets) + [EOS]
        yield np.array(inputs_with_eos), np.array(targets_with_eos)

