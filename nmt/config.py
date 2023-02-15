"""
Configuration dataclasses for NMT model and training.

This module provides type-safe configuration objects for various aspects
of the machine translation system.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """
    Configuration for the NMT model architecture.
    
    Attributes:
        input_vocab_size: Size of input vocabulary
        target_vocab_size: Size of target vocabulary
        d_model: Embedding dimension and LSTM hidden size
        n_encoder_layers: Number of LSTM layers in encoder
        n_decoder_layers: Number of LSTM layers in decoder
        n_attention_heads: Number of attention heads
        attention_dropout: Dropout rate for attention layer (0.0 to 1.0)
        mode: Model mode ('train', 'eval', or 'predict')
    """
    input_vocab_size: int = 33300
    target_vocab_size: int = 33300
    d_model: int = 1024
    n_encoder_layers: int = 2
    n_decoder_layers: int = 2
    n_attention_heads: int = 4
    attention_dropout: float = 0.0
    mode: str = 'train'
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.input_vocab_size <= 0:
            raise ValueError(f"input_vocab_size must be positive, got {self.input_vocab_size}")
        if self.target_vocab_size <= 0:
            raise ValueError(f"target_vocab_size must be positive, got {self.target_vocab_size}")
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.n_encoder_layers <= 0:
            raise ValueError(f"n_encoder_layers must be positive, got {self.n_encoder_layers}")
        if self.n_decoder_layers <= 0:
            raise ValueError(f"n_decoder_layers must be positive, got {self.n_decoder_layers}")
        if self.n_attention_heads <= 0:
            raise ValueError(f"n_attention_heads must be positive, got {self.n_attention_heads}")
        if not 0.0 <= self.attention_dropout <= 1.0:
            raise ValueError(f"attention_dropout must be in [0.0, 1.0], got {self.attention_dropout}")
        if self.mode not in ['train', 'eval', 'predict']:
            raise ValueError(f"mode must be 'train', 'eval', or 'predict', got {self.mode}")
    
    def to_dict(self):
        """Convert to dictionary for passing to model functions."""
        return {
            'input_vocab_size': self.input_vocab_size,
            'target_vocab_size': self.target_vocab_size,
            'd_model': self.d_model,
            'n_encoder_layers': self.n_encoder_layers,
            'n_decoder_layers': self.n_decoder_layers,
            'n_attention_heads': self.n_attention_heads,
            'attention_dropout': self.attention_dropout,
            'mode': self.mode
        }


@dataclass
class DataConfig:
    """
    Configuration for data loading and preprocessing.
    
    Attributes:
        data_dir: Directory containing the dataset
        vocab_file: Vocabulary filename
        vocab_dir: Vocabulary directory
        max_length: Maximum sequence length
        boundaries: Bucketing boundaries for sequence lengths
        batch_sizes: Batch sizes for each bucket
    """
    data_dir: str = './data/'
    vocab_file: str = 'ende_32k.subword'
    vocab_dir: str = 'data/'
    max_length: int = 512
    boundaries: list = field(default_factory=lambda: [8, 16, 32, 64, 128, 256, 512])
    batch_sizes: list = field(default_factory=lambda: [256, 128, 64, 32, 16, 8, 4, 2])
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")
        if len(self.boundaries) != len(self.batch_sizes) - 1:
            raise ValueError(
                f"batch_sizes must have one more element than boundaries. "
                f"Got {len(self.boundaries)} boundaries and {len(self.batch_sizes)} batch_sizes"
            )


@dataclass
class TrainingConfig:
    """
    Configuration for model training.
    
    Attributes:
        learning_rate: Maximum learning rate
        warmup_steps: Number of warmup steps for learning rate schedule
        checkpoint_freq: Steps between model checkpoints
        n_steps: Total number of training steps
        output_dir: Directory for saving checkpoints
    """
    learning_rate: float = 0.01
    warmup_steps: int = 1000
    checkpoint_freq: int = 10
    n_steps: int = 10
    output_dir: str = 'output_dir/'
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {self.warmup_steps}")
        if self.checkpoint_freq <= 0:
            raise ValueError(f"checkpoint_freq must be positive, got {self.checkpoint_freq}")
        if self.n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {self.n_steps}")


@dataclass
class InferenceConfig:
    """
    Configuration for model inference.
    
    Attributes:
        model_path: Path to saved model weights
        temperature: Sampling temperature (0.0 = greedy, higher = more random)
        vocab_file: Vocabulary filename
        vocab_dir: Vocabulary directory
        method: Decoding method ('greedy', 'sampling', or 'mbr')
        n_samples: Number of samples for MBR decoding
    """
    model_path: str = "model.pkl.gz"
    temperature: float = 0.0
    vocab_file: Optional[str] = None
    vocab_dir: Optional[str] = None
    method: str = 'greedy'
    n_samples: int = 10
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.temperature < 0:
            raise ValueError(f"temperature must be non-negative, got {self.temperature}")
        if self.method not in ['greedy', 'sampling', 'mbr']:
            raise ValueError(f"method must be 'greedy', 'sampling', or 'mbr', got {self.method}")
        if self.n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {self.n_samples}")

