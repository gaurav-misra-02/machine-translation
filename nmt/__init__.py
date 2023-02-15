"""
Neural Machine Translation (NMT) Package

A complete implementation of English-to-German neural machine translation
using LSTM networks with attention mechanism.

Author: Gaurav Misra
"""

__version__ = "1.0.0"

# Data loading and preprocessing
from .data import (
    load_data,
    prepare_data_pipeline,
    get_data_pipeline
)

# Model architecture
from .model import (
    NMTAttn,
    input_encoder_fn,
    pre_attention_decoder_fn,
    prepare_attention_input
)

# Training
from .training import (
    create_train_task,
    create_eval_task,
    train_model,
    train_from_config
)

# Inference and decoding
from .inference import (
    # Decoding functions
    next_symbol,
    sampling_decode,
    greedy_decode,
    mbr_decode,
    
    # Sampling utilities
    generate_samples,
    
    # Similarity metrics
    jaccard_similarity,
    rouge1_similarity,
    average_overlap,
    weighted_avg_overlap,
    
    # Convenience functions
    load_model,
    translate_sentence,
    mbr_translate
)

# Utilities
from .utils import (
    tokenize,
    detokenize,
    append_eos,
    VOCAB_FILE,
    VOCAB_DIR,
    EOS
)

# Configuration
from .config import (
    ModelConfig,
    DataConfig,
    TrainingConfig,
    InferenceConfig
)

__all__ = [
    # Data
    'load_data',
    'prepare_data_pipeline',
    'get_data_pipeline',
    
    # Model
    'NMTAttn',
    'input_encoder_fn',
    'pre_attention_decoder_fn',
    'prepare_attention_input',
    
    # Training
    'create_train_task',
    'create_eval_task',
    'train_model',
    'train_from_config',
    
    # Inference
    'next_symbol',
    'sampling_decode',
    'greedy_decode',
    'mbr_decode',
    'generate_samples',
    'jaccard_similarity',
    'rouge1_similarity',
    'average_overlap',
    'weighted_avg_overlap',
    'load_model',
    'translate_sentence',
    'mbr_translate',
    
    # Utils
    'tokenize',
    'detokenize',
    'append_eos',
    'VOCAB_FILE',
    'VOCAB_DIR',
    'EOS',
    
    # Config
    'ModelConfig',
    'DataConfig',
    'TrainingConfig',
    'InferenceConfig',
]

