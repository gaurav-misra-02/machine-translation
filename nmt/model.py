"""
Neural Machine Translation model architecture.

This module implements the encoder-decoder architecture with attention mechanism.
The model uses LSTMs for sequential processing and scaled dot-product attention
to handle long-range dependencies.
"""

from typing import Tuple
from trax import layers as tl
from trax.fastmath import numpy as fastnp

# Model default configuration constants
DEFAULT_VOCAB_SIZE = 33300
DEFAULT_D_MODEL = 1024
DEFAULT_N_ENCODER_LAYERS = 2
DEFAULT_N_DECODER_LAYERS = 2
DEFAULT_N_ATTENTION_HEADS = 4
DEFAULT_ATTENTION_DROPOUT = 0.0


def input_encoder_fn(input_vocab_size: int, 
                      d_model: int, 
                      n_encoder_layers: int) -> tl.Serial:
    """
    Build the input encoder network.
    
    Takes input tokens, converts to embeddings, and processes through stacked LSTMs.
    The output activations serve as keys and values for the attention mechanism.
    
    Args:
        input_vocab_size: Size of input vocabulary (must be positive)
        d_model: Embedding dimension and LSTM hidden size (must be positive)
        n_encoder_layers: Number of LSTM layers to stack (must be positive)
        
    Returns:
        The input encoder network
    
    Raises:
        ValueError: If any parameter is not positive
    """
    if input_vocab_size <= 0:
        raise ValueError(f"input_vocab_size must be positive, got {input_vocab_size}")
    if d_model <= 0:
        raise ValueError(f"d_model must be positive, got {d_model}")
    if n_encoder_layers <= 0:
        raise ValueError(f"n_encoder_layers must be positive, got {n_encoder_layers}")
    input_encoder = tl.Serial( 
        # Token embedding layer
        tl.Embedding(input_vocab_size, d_model),
        
        # Stack of LSTM layers
        [tl.LSTM(d_model) for _ in range(n_encoder_layers)]
    )

    return input_encoder


def pre_attention_decoder_fn(mode: str, 
                             target_vocab_size: int, 
                             d_model: int) -> tl.Serial:
    """
    Build the pre-attention decoder network.
    
    Processes target tokens before attention. ShiftRight enables teacher forcing
    during training and provides start-of-sequence token during inference.
    
    Args:
        mode: 'train' or 'eval' mode
        target_vocab_size: Size of target vocabulary (must be positive)
        d_model: Embedding dimension and LSTM hidden size (must be positive)
        
    Returns:
        The pre-attention decoder network
    
    Raises:
        ValueError: If target_vocab_size or d_model is not positive
    """
    if target_vocab_size <= 0:
        raise ValueError(f"target_vocab_size must be positive, got {target_vocab_size}")
    if d_model <= 0:
        raise ValueError(f"d_model must be positive, got {d_model}")
    pre_attention_decoder = tl.Serial(
        # Shift right for autoregressive structure
        tl.ShiftRight(),

        # Token embedding layer
        tl.Embedding(target_vocab_size, d_model),

        # LSTM processing
        tl.LSTM(d_model)
    )
    
    return pre_attention_decoder


def prepare_attention_input(encoder_activations: fastnp.ndarray, 
                           decoder_activations: fastnp.ndarray, 
                           inputs: fastnp.ndarray) -> Tuple[fastnp.ndarray, fastnp.ndarray, 
                                                            fastnp.ndarray, fastnp.ndarray]:
    """
    Prepare queries, keys, values and mask for the attention mechanism.
    
    Transforms encoder and decoder activations into the format expected by
    the attention layer, including proper masking for padding tokens.
    
    Args:
        encoder_activations: Encoder output 
            Shape: (batch_size, padded_input_length, d_model)
        decoder_activations: Pre-attention decoder output
            Shape: (batch_size, padded_input_length, d_model)
        inputs: Input token IDs
            Shape: (batch_size, padded_input_length)
    
    Returns:
        (queries, keys, values, mask) ready for attention layer
    """
    # Keys and values come from encoder (source sentence)
    keys = encoder_activations
    values = encoder_activations
    
    # Queries come from decoder (target sentence being generated)
    queries = decoder_activations
    
    # Create mask: True for real tokens, False for padding
    # Padding tokens have value 0, real tokens are positive integers
    mask = inputs > 0
    
    # Reshape mask for multi-headed attention
    # Add axes for: attention heads, decoder sequence length
    mask = fastnp.reshape(mask, (mask.shape[0], 1, 1, mask.shape[1]))
    
    # Broadcast to final shape: [batch_size, attention_heads, decoder_len, encoder_len]
    mask = mask + fastnp.zeros((1, 1, decoder_activations.shape[1], 1))
    
    return queries, keys, values, mask


def NMTAttn(input_vocab_size: int = DEFAULT_VOCAB_SIZE,
            target_vocab_size: int = DEFAULT_VOCAB_SIZE,
            d_model: int = DEFAULT_D_MODEL,
            n_encoder_layers: int = DEFAULT_N_ENCODER_LAYERS,
            n_decoder_layers: int = DEFAULT_N_DECODER_LAYERS,
            n_attention_heads: int = DEFAULT_N_ATTENTION_HEADS,
            attention_dropout: float = DEFAULT_ATTENTION_DROPOUT,
            mode: str = 'train') -> tl.Serial:
    """
    Build the complete Neural Machine Translation model with attention.
    
    This combines the encoder, decoder, and attention mechanism into a single network.
    The attention layer is wrapped in a residual connection to help with gradient flow.

    Args:
        input_vocab_size: Vocab size of the input
        target_vocab_size: Vocab size of the target
        d_model: Depth of embedding
        n_encoder_layers: Number of LSTM layers in encoder
        n_decoder_layers: Number of LSTM layers in decoder after attention
        n_attention_heads: Number of attention heads
        attention_dropout: Dropout rate for attention layer (0.0 to 1.0)
        mode: 'train', 'eval' or 'predict'

    Returns:
        Complete LSTM sequence-to-sequence model with attention
    
    Raises:
        ValueError: If any parameter is invalid
    """
    # Validate parameters
    if input_vocab_size <= 0 or target_vocab_size <= 0:
        raise ValueError("Vocabulary sizes must be positive")
    if d_model <= 0:
        raise ValueError(f"d_model must be positive, got {d_model}")
    if n_encoder_layers <= 0 or n_decoder_layers <= 0:
        raise ValueError("Number of layers must be positive")
    if n_attention_heads <= 0:
        raise ValueError(f"n_attention_heads must be positive, got {n_attention_heads}")
    if not 0.0 <= attention_dropout <= 1.0:
        raise ValueError(f"attention_dropout must be in [0.0, 1.0], got {attention_dropout}")
    if mode not in ['train', 'eval', 'predict']:
        raise ValueError(f"mode must be 'train', 'eval', or 'predict', got {mode}")
    # Build encoder and pre-attention decoder
    input_encoder = input_encoder_fn(input_vocab_size, d_model, n_encoder_layers)
    pre_attention_decoder = pre_attention_decoder_fn(mode, target_vocab_size, d_model)
    
    # Build the complete model
    model = tl.Serial(
        # Copy input and target tokens (needed by both encoder and decoder)
        tl.Select([0, 1, 0, 1]),
        
        # Run encoder and pre-attention decoder in parallel
        tl.Parallel(input_encoder, pre_attention_decoder),
        
        # Prepare attention inputs
        tl.Fn('PrepareAttentionInput', prepare_attention_input, n_out=4),
        
        # Attention with residual connection
        tl.Residual(
            tl.AttentionQKV(d_model, n_heads=n_attention_heads, 
                          dropout=attention_dropout, mode=None)
        ),
        
        # Drop attention mask (no longer needed)
        tl.Select([0, 2]),
        
        # Post-attention decoder layers
        [tl.LSTM(d_model) for _ in range(n_decoder_layers)],
        
        # Output projection to vocabulary size
        tl.Dense(target_vocab_size),
        
        # Log-softmax for probabilities
        tl.LogSoftmax()
    )
    
    return model

