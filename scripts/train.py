#!/usr/bin/env python
"""
Training script for Neural Machine Translation model.

This script trains an NMT model from scratch or continues training
from a checkpoint.

Usage:
    python scripts/train.py --steps 1000 --output_dir checkpoints/
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nmt import get_data_pipeline, train_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


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
    
    logger.info("=" * 70)
    logger.info("Neural Machine Translation - Training")
    logger.info("=" * 70)
    logger.info("\nConfiguration:")
    logger.info(f"  Training steps: {args.steps}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Max sequence length: {args.max_length}")
    logger.info(f"  Model dimension: {args.d_model}")
    logger.info(f"  Encoder layers: {args.n_encoder_layers}")
    logger.info(f"  Decoder layers: {args.n_decoder_layers}")
    logger.info(f"  Attention heads: {args.n_attention_heads}")
    
    # Validate arguments
    if args.steps <= 0:
        logger.error(f"Training steps must be positive, got {args.steps}")
        sys.exit(1)
    
    if args.max_length <= 0:
        logger.error(f"Max length must be positive, got {args.max_length}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output_dir)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory created/verified: {args.output_dir}")
    except Exception as e:
        logger.error(f"Failed to create output directory: {e}")
        sys.exit(1)
    
    # Load data
    logger.info("Loading data pipeline...")
    try:
        train_batch_stream, eval_batch_stream = get_data_pipeline(
            max_length=args.max_length
        )
        logger.info("Data loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)
    
    # Model configuration
    model_config = {
        'd_model': args.d_model,
        'n_encoder_layers': args.n_encoder_layers,
        'n_decoder_layers': args.n_decoder_layers,
        'n_attention_heads': args.n_attention_heads,
    }
    
    # Train model
    logger.info(f"Starting training for {args.steps} steps...")
    logger.info("-" * 70)
    
    try:
        training_loop = train_model(
            train_batch_stream,
            eval_batch_stream,
            n_steps=args.steps,
            output_dir=args.output_dir,
            model_config=model_config
        )
        
        logger.info("-" * 70)
        logger.info("Training complete!")
        logger.info(f"Model checkpoints saved to: {args.output_dir}")
    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

