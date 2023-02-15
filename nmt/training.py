"""
Training configuration and loop for NMT model.

This module provides functions to configure and run the training process,
including loss functions, optimizers, learning rate schedules, and evaluation.
"""

from typing import Dict, Generator, Optional
import trax
from trax import layers as tl
from trax.supervised import training

from .model import NMTAttn

# Training default constants
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_WARMUP_STEPS = 1000
DEFAULT_CHECKPOINT_FREQ = 10
DEFAULT_N_STEPS = 10
DEFAULT_OUTPUT_DIR = 'output_dir/'


def create_train_task(train_batch_stream: Generator, 
                     learning_rate: float = DEFAULT_LEARNING_RATE, 
                     warmup_steps: int = DEFAULT_WARMUP_STEPS, 
                     checkpoint_freq: int = DEFAULT_CHECKPOINT_FREQ) -> training.TrainTask:
    """
    Create the training task configuration.
    
    Uses Adam optimizer with a learning rate schedule that includes warmup.
    The warmup phase helps stabilize training in the early stages.
    
    Args:
        train_batch_stream: Generator for training batches
        learning_rate: Maximum learning rate (must be positive)
        warmup_steps: Number of warmup steps (must be non-negative)
        checkpoint_freq: Steps between checkpoints (must be positive)
        
    Returns:
        Configured training task
    
    Raises:
        ValueError: If parameters are invalid
    """
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")
    if checkpoint_freq <= 0:
        raise ValueError(f"checkpoint_freq must be positive, got {checkpoint_freq}")
    return training.TrainTask(
        labeled_data=train_batch_stream,
        loss_layer=tl.CrossEntropyLoss(),
        optimizer=trax.optimizers.adam.Adam(learning_rate),
        lr_schedule=trax.lr.warmup_and_rsqrt_decay(warmup_steps, learning_rate),
        n_steps_per_checkpoint=checkpoint_freq
    )


def create_eval_task(eval_batch_stream: Generator) -> training.EvalTask:
    """
    Create the evaluation task configuration.
    
    Tracks both loss and accuracy during training to monitor model performance.
    
    Args:
        eval_batch_stream: Generator for evaluation batches
        
    Returns:
        Configured evaluation task
    """
    return training.EvalTask(
        labeled_data=eval_batch_stream,
        metrics=[tl.CrossEntropyLoss(), tl.Accuracy()]
    )


def train_model(train_batch_stream: Generator, 
                eval_batch_stream: Generator, 
                n_steps: int = DEFAULT_N_STEPS, 
                output_dir: str = DEFAULT_OUTPUT_DIR,
                model_config: Optional[Dict] = None) -> training.Loop:
    """
    Train the NMT model.
    
    Sets up the complete training loop including model initialization,
    training task, evaluation task, and training execution.
    
    Args:
        train_batch_stream: Training data batch generator
        eval_batch_stream: Evaluation data batch generator
        n_steps: Number of training steps (must be positive)
        output_dir: Directory for saving checkpoints
        model_config: Model configuration parameters (uses defaults if None)
        
    Returns:
        The training loop (can be used to continue training)
    
    Raises:
        ValueError: If n_steps is not positive
    """
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    # Create tasks
    train_task = create_train_task(train_batch_stream)
    eval_task = create_eval_task(eval_batch_stream)
    
    # Initialize model with config or defaults
    if model_config is None:
        model_config = {}
    model = NMTAttn(mode='train', **model_config)
    
    # Create and run training loop
    training_loop = training.Loop(
        model,
        train_task,
        eval_tasks=[eval_task],
        output_dir=output_dir
    )
    
    training_loop.run(n_steps)
    
    return training_loop


def train_from_config(data_pipeline: tuple, config: Dict) -> training.Loop:
    """
    Train model from a configuration dictionary.
    
    Convenience function that accepts all training parameters in a single config dict.
    
    Args:
        data_pipeline: (train_batch_stream, eval_batch_stream)
        config: Configuration dictionary with optional keys:
            - n_steps: Training steps (default: 10)
            - output_dir: Checkpoint directory (default: 'output_dir/')
            - model: Model configuration dict
            - learning_rate: Learning rate
            - warmup_steps: Warmup steps
            
    Returns:
        The training loop
    
    Raises:
        ValueError: If data_pipeline format is invalid
    """
    if not isinstance(data_pipeline, tuple) or len(data_pipeline) != 2:
        raise ValueError("data_pipeline must be a tuple of (train_stream, eval_stream)")
    train_batch_stream, eval_batch_stream = data_pipeline
    
    return train_model(
        train_batch_stream,
        eval_batch_stream,
        n_steps=config.get('n_steps', 10),
        output_dir=config.get('output_dir', 'output_dir/'),
        model_config=config.get('model', None)
    )

