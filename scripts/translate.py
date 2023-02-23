#!/usr/bin/env python
"""
Translation script for Neural Machine Translation model.

This script translates English sentences to German using a trained model.
Supports multiple decoding strategies: greedy, sampling, and MBR.

Usage:
    python scripts/translate.py --sentence "Hello world" --method greedy
    python scripts/translate.py --sentence "I love AI" --method mbr --samples 10
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nmt import (
    load_model, 
    sampling_decode, 
    mbr_decode, 
    average_overlap, 
    rouge1_similarity,
    jaccard_similarity
)

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
        description='Translate English to German using NMT model'
    )
    
    # Input
    parser.add_argument(
        '--sentence',
        type=str,
        required=True,
        help='English sentence to translate'
    )
    
    # Model
    parser.add_argument(
        '--model_path',
        type=str,
        default='model.pkl.gz',
        help='Path to model weights (default: model.pkl.gz)'
    )
    
    # Decoding method
    parser.add_argument(
        '--method',
        type=str,
        choices=['greedy', 'sampling', 'mbr'],
        default='greedy',
        help='Decoding method (default: greedy)'
    )
    
    # Sampling parameters
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.6,
        help='Sampling temperature (default: 0.6, only for sampling/mbr)'
    )
    
    # MBR parameters
    parser.add_argument(
        '--samples',
        type=int,
        default=10,
        help='Number of samples for MBR decoding (default: 10)'
    )
    parser.add_argument(
        '--similarity',
        type=str,
        choices=['rouge1', 'jaccard'],
        default='rouge1',
        help='Similarity function for MBR (default: rouge1)'
    )
    
    return parser.parse_args()


def main():
    """Main translation function."""
    args = parse_args()
    
    logger.info("=" * 70)
    logger.info("Neural Machine Translation - Inference")
    logger.info("=" * 70)
    logger.info(f"\nInput (English): {args.sentence}")
    logger.info(f"Decoding method: {args.method}")
    
    # Validate input
    if not args.sentence or not args.sentence.strip():
        logger.error("Input sentence cannot be empty")
        sys.exit(1)
    
    # Check if model file exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model file not found: {args.model_path}")
        logger.info("\nMake sure you have trained a model first:")
        logger.info("  python scripts/train.py --steps 1000")
        sys.exit(1)
    
    # Load model
    logger.info(f"\nLoading model from {args.model_path}...")
    try:
        model = load_model(args.model_path)
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)
    
    # Validate parameters
    if args.temperature < 0:
        logger.error(f"Temperature must be non-negative, got {args.temperature}")
        sys.exit(1)
    
    if args.samples <= 0:
        logger.error(f"Number of samples must be positive, got {args.samples}")
        sys.exit(1)
    
    # Translate
    logger.info(f"\nTranslating...")
    logger.info("-" * 70)
    
    try:
        if args.method == 'greedy':
            # Greedy decoding (temperature = 0)
            _, _, translation = sampling_decode(
                args.sentence, 
                model, 
                temperature=0.0
            )
            logger.info(f"\nOutput (German): {translation}")
            
        elif args.method == 'sampling':
            # Sampling with temperature
            logger.info(f"Temperature: {args.temperature}")
            _, _, translation = sampling_decode(
                args.sentence, 
                model, 
                temperature=args.temperature
            )
            logger.info(f"\nOutput (German): {translation}")
            
        elif args.method == 'mbr':
            # MBR decoding
            logger.info(f"Generating {args.samples} candidates...")
            logger.info(f"Similarity metric: {args.similarity}")
            
            similarity_fn = (rouge1_similarity if args.similarity == 'rouge1' 
                           else jaccard_similarity)
            
            translation, best_idx, scores = mbr_decode(
                args.sentence,
                n_samples=args.samples,
                score_fn=average_overlap,
                similarity_fn=similarity_fn,
                model=model,
                temperature=args.temperature
            )
            
            logger.info(f"\nBest candidate: #{best_idx} (score: {scores[best_idx]:.4f})")
            logger.info(f"Output (German): {translation}")
        
        logger.info("-" * 70)
        
    except KeyboardInterrupt:
        logger.warning("\nTranslation interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

