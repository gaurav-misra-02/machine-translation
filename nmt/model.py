"""
Neural Machine Translation model architecture.

This module implements the encoder-decoder architecture with attention mechanism.
The model uses LSTMs for sequential processing and scaled dot-product attention
to handle long-range dependencies.
"""

from trax import layers as tl
from trax.fastmath import numpy as fastnp


def input_encoder_fn(input_vocab_size, d_model, n_encoder_layers):
    """
    Build the input encoder network.
    
    Takes input tokens, converts to embeddings, and processes through stacked LSTMs.
    The output activations serve as keys and values for the attention mechanism.
    
    Args:
        input_vocab_size (int): Size of input vocabulary
        d_model (int): Embedding dimension and LSTM hidden size
        n_encoder_layers (int): Number of LSTM layers to stack
        
    Returns:
        tl.Serial: The input encoder network
    """
    input_encoder = tl.Serial( 
        # Token embedding layer
        tl.Embedding(input_vocab_size, d_model),
        
        # Stack of LSTM layers
        [tl.LSTM(d_model) for _ in range(n_encoder_layers)]
    )

    return input_encoder


def pre_attention_decoder_fn(mode, target_vocab_size, d_model):
    """
    Build the pre-attention decoder network.
    
    Processes target tokens before attention. ShiftRight enables teacher forcing
    during training and provides start-of-sequence token during inference.
    
    Args:
        mode (str): 'train' or 'eval' mode
        target_vocab_size (int): Size of target vocabulary
        d_model (int): Embedding dimension and LSTM hidden size
        
    Returns:
        tl.Serial: The pre-attention decoder network
    """
    pre_attention_decoder = tl.Serial(
        # Shift right for autoregressive structure
        tl.ShiftRight(),

        # Token embedding layer
        tl.Embedding(target_vocab_size, d_model),

        # LSTM processing
        tl.LSTM(d_model)
    )
    
    return pre_attention_decoder


def prepare_attention_input(encoder_activations, decoder_activations, inputs):
    """
    Prepare queries, keys, values and mask for the attention mechanism.
    
    Transforms encoder and decoder activations into the format expected by
    the attention layer, including proper masking for padding tokens.
    
    Args:
        encoder_activations (fastnp.array): Encoder output 
            Shape: (batch_size, padded_input_length, d_model)
        decoder_activations (fastnp.array): Pre-attention decoder output
            Shape: (batch_size, padded_input_length, d_model)
        inputs (fastnp.array): Input token IDs
            Shape: (batch_size, padded_input_length)
    
    Returns:
        tuple: (queries, keys, values, mask) ready for attention layer
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


def NMTAttn(input_vocab_size=33300,
            target_vocab_size=33300,
            d_model=1024,
            n_encoder_layers=2,
            n_decoder_layers=2,
            n_attention_heads=4,
            attention_dropout=0.0,
            mode='train'):
    """
    Build the complete Neural Machine Translation model with attention.
    
    This combines the encoder, decoder, and attention mechanism into a single network.
    The attention layer is wrapped in a residual connection to help with gradient flow.

    Args:
        input_vocab_size (int): Vocab size of the input (default: 33300)
        target_vocab_size (int): Vocab size of the target (default: 33300)
        d_model (int): Depth of embedding (default: 1024)
        n_encoder_layers (int): Number of LSTM layers in encoder (default: 2)
        n_decoder_layers (int): Number of LSTM layers in decoder after attention (default: 2)
        n_attention_heads (int): Number of attention heads (default: 4)
        attention_dropout (float): Dropout rate for attention layer (default: 0.0)
        mode (str): 'train', 'eval' or 'predict' (default: 'train')

    Returns:
        tl.Serial: Complete LSTM sequence-to-sequence model with attention
    """
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

