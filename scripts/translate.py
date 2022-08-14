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
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nmt import (
    load_model, 
    sampling_decode, 
    mbr_decode, 
    average_overlap, 
    rouge1_similarity
)


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
    
    print("=" * 70)
    print("Neural Machine Translation - Inference")
    print("=" * 70)
    print(f"\nInput (English): {args.sentence}")
    print(f"Decoding method: {args.method}")
    
    # Load model
    print(f"\nLoading model from {args.model_path}...")
    try:
        model = load_model(args.model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nMake sure you have trained a model first:")
        print("  python scripts/train.py --steps 1000")
        sys.exit(1)
    
    # Translate
    print(f"\nTranslating...")
    print("-" * 70)
    
    if args.method == 'greedy':
        # Greedy decoding (temperature = 0)
        _, _, translation = sampling_decode(
            args.sentence, 
            model, 
            temperature=0.0
        )
        print(f"\nOutput (German): {translation}")
        
    elif args.method == 'sampling':
        # Sampling with temperature
        print(f"Temperature: {args.temperature}")
        _, _, translation = sampling_decode(
            args.sentence, 
            model, 
            temperature=args.temperature
        )
        print(f"\nOutput (German): {translation}")
        
    elif args.method == 'mbr':
        # MBR decoding
        print(f"Generating {args.samples} candidates...")
        print(f"Similarity metric: {args.similarity}")
        
        similarity_fn = rouge1_similarity if args.similarity == 'rouge1' else None
        
        translation, best_idx, scores = mbr_decode(
            args.sentence,
            n_samples=args.samples,
            score_fn=average_overlap,
            similarity_fn=similarity_fn,
            model=model,
            temperature=args.temperature
        )
        
        print(f"\nBest candidate: #{best_idx} (score: {scores[best_idx]:.4f})")
        print(f"Output (German): {translation}")
    
    print("-" * 70)
    print()


if __name__ == "__main__":
    main()

