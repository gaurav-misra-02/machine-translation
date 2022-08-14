"""
Training configuration and loop for NMT model.

This module provides functions to configure and run the training process,
including loss functions, optimizers, learning rate schedules, and evaluation.
"""

import trax
from trax import layers as tl
from trax.supervised import training

from .model import NMTAttn


def create_train_task(train_batch_stream, learning_rate=0.01, 
                     warmup_steps=1000, checkpoint_freq=10):
    """
    Create the training task configuration.
    
    Uses Adam optimizer with a learning rate schedule that includes warmup.
    The warmup phase helps stabilize training in the early stages.
    
    Args:
        train_batch_stream: Generator for training batches
        learning_rate (float): Maximum learning rate (default: 0.01)
        warmup_steps (int): Number of warmup steps (default: 1000)
        checkpoint_freq (int): Steps between checkpoints (default: 10)
        
    Returns:
        training.TrainTask: Configured training task
    """
    return training.TrainTask(
        labeled_data=train_batch_stream,
        loss_layer=tl.CrossEntropyLoss(),
        optimizer=trax.optimizers.adam.Adam(learning_rate),
        lr_schedule=trax.lr.warmup_and_rsqrt_decay(warmup_steps, learning_rate),
        n_steps_per_checkpoint=checkpoint_freq
    )


def create_eval_task(eval_batch_stream):
    """
    Create the evaluation task configuration.
    
    Tracks both loss and accuracy during training to monitor model performance.
    
    Args:
        eval_batch_stream: Generator for evaluation batches
        
    Returns:
        training.EvalTask: Configured evaluation task
    """
    return training.EvalTask(
        labeled_data=eval_batch_stream,
        metrics=[tl.CrossEntropyLoss(), tl.Accuracy()]
    )


def train_model(train_batch_stream, eval_batch_stream, 
                n_steps=10, output_dir='output_dir/',
                model_config=None):
    """
    Train the NMT model.
    
    Sets up the complete training loop including model initialization,
    training task, evaluation task, and training execution.
    
    Args:
        train_batch_stream: Training data batch generator
        eval_batch_stream: Evaluation data batch generator
        n_steps (int): Number of training steps (default: 10)
        output_dir (str): Directory for saving checkpoints (default: 'output_dir/')
        model_config (dict): Model configuration parameters (default: None uses defaults)
        
    Returns:
        training.Loop: The training loop (can be used to continue training)
    """
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


def train_from_config(data_pipeline, config):
    """
    Train model from a configuration dictionary.
    
    Convenience function that accepts all training parameters in a single config dict.
    
    Args:
        data_pipeline (tuple): (train_batch_stream, eval_batch_stream)
        config (dict): Configuration with keys:
            - n_steps (int): Training steps
            - output_dir (str): Checkpoint directory
            - model (dict): Model configuration
            - learning_rate (float): Learning rate
            - warmup_steps (int): Warmup steps
            
    Returns:
        training.Loop: The training loop
    """
    train_batch_stream, eval_batch_stream = data_pipeline
    
    return train_model(
        train_batch_stream,
        eval_batch_stream,
        n_steps=config.get('n_steps', 10),
        output_dir=config.get('output_dir', 'output_dir/'),
        model_config=config.get('model', None)
    )

