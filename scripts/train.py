#!/usr/bin/env python
"""
Training script for Neural Machine Translation model.

This script trains an NMT model from scratch or continues training
from a checkpoint.

Usage:
    python scripts/train.py --steps 1000 --output_dir checkpoints/
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nmt import get_data_pipeline, train_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Neural Machine Translation model'
    )
    
    # Training parameters
    parser.add_argument(
        '--steps', 
        type=int, 
        default=100,
        help='Number of training steps (default: 100)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output_dir/',
        help='Directory for checkpoints (default: output_dir/)'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help='Maximum sequence length (default: 512)'
    )
    
    # Model parameters
    parser.add_argument(
        '--d_model',
        type=int,
        default=1024,
        help='Model dimension (default: 1024)'
    )
    parser.add_argument(
        '--n_encoder_layers',
        type=int,
        default=2,
        help='Number of encoder LSTM layers (default: 2)'
    )
    parser.add_argument(
        '--n_decoder_layers',
        type=int,
        default=2,
        help='Number of decoder LSTM layers (default: 2)'
    )
    parser.add_argument(
        '--n_attention_heads',
        type=int,
        default=4,
        help='Number of attention heads (default: 4)'
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    print("=" * 70)
    print("Neural Machine Translation - Training")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Training steps: {args.steps}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Max sequence length: {args.max_length}")
    print(f"  Model dimension: {args.d_model}")
    print(f"  Encoder layers: {args.n_encoder_layers}")
    print(f"  Decoder layers: {args.n_decoder_layers}")
    print(f"  Attention heads: {args.n_attention_heads}")
    print()
    
    # Load data
    print("Loading data...")
    train_batch_stream, eval_batch_stream = get_data_pipeline(
        max_length=args.max_length
    )
    print("Data loaded successfully!")
    print()
    
    # Model configuration
    model_config = {
        'd_model': args.d_model,
        'n_encoder_layers': args.n_encoder_layers,
        'n_decoder_layers': args.n_decoder_layers,
        'n_attention_heads': args.n_attention_heads,
    }
    
    # Train model
    print(f"Starting training for {args.steps} steps...")
    print("-" * 70)
    
    training_loop = train_model(
        train_batch_stream,
        eval_batch_stream,
        n_steps=args.steps,
        output_dir=args.output_dir,
        model_config=model_config
    )
    
    print("-" * 70)
    print("\nTraining complete!")
    print(f"Model checkpoints saved to: {args.output_dir}")
    print()


if __name__ == "__main__":
    main()

