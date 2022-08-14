"""
Inference and decoding strategies for NMT model.

This module implements multiple decoding approaches:
- Greedy decoding (fast, deterministic)
- Sampling decoding (diverse, controllable randomness)
- Minimum Bayes Risk (MBR) decoding (high quality, slower)
"""

import numpy as np
from collections import Counter
from trax import layers as tl

from .utils import tokenize, detokenize, EOS, VOCAB_FILE, VOCAB_DIR
from .model import NMTAttn


def next_symbol(model, input_tokens, cur_output_tokens, temperature):
    """
    Predict the next token in the translation.
    
    Handles the autoregressive generation process. Pads the current output
    to the next power of 2 for efficiency and gets the model's prediction.

    Args:
        model (tl.Serial): The NMT model
        input_tokens (np.ndarray): Tokenized input sentence (1 x n_tokens)
        cur_output_tokens (list): Previously generated tokens
        temperature (float): Sampling temperature (0.0 = greedy, 1.0 = random)

    Returns:
        tuple: (next_token_index, log_probability)
    """
    token_length = len(cur_output_tokens)
    
    # Pad to next power of 2
    padded_length = 2**int(np.ceil(np.log2(token_length + 1)))
    padded = cur_output_tokens + [0] * (padded_length - token_length)
    
    # Add batch dimension
    padded_with_batch = np.reshape(np.array(padded), (1, padded_length))
    
    # Get model prediction
    output, _ = model((input_tokens, padded_with_batch))
    
    # Extract log probabilities for the last token
    log_probs = output[0, -1, :]
    
    # Sample next token (temperature=0 gives greedy decoding)
    symbol = int(tl.logsoftmax_sample(log_probs, temperature))
    
    return symbol, float(log_probs[symbol])


def sampling_decode(input_sentence, model=None, temperature=0.0, 
                   vocab_file=None, vocab_dir=None):
    """
    Translate a sentence using sampling-based decoding.
    
    Temperature controls the randomness:
    - 0.0: Greedy decoding (always pick most probable)
    - 1.0: Sample from full distribution (most random)
    - 0.3-0.7: Sweet spot for diverse but reasonable translations

    Args:
        input_sentence (str): Sentence to translate
        model (tl.Serial): The NMT model
        temperature (float): Sampling temperature (default: 0.0)
        vocab_file (str): Vocabulary filename (default: from utils)
        vocab_dir (str): Path to vocabulary file (default: from utils)

    Returns:
        tuple: (token_list, log_probability, translated_sentence)
    """
    if vocab_file is None:
        vocab_file = VOCAB_FILE
    if vocab_dir is None:
        vocab_dir = VOCAB_DIR
        
    # Tokenize input
    input_tokens = tokenize(input_sentence, vocab_file, vocab_dir)
    
    # Generate output tokens one by one
    cur_output_tokens = []
    cur_output = 0
    
    # Continue until EOS token
    while cur_output != EOS:
        cur_output, log_prob = next_symbol(model, input_tokens, cur_output_tokens, temperature)
        cur_output_tokens.append(cur_output)
    
    # Detokenize to get final sentence
    sentence = detokenize(cur_output_tokens, vocab_file, vocab_dir)
    
    return cur_output_tokens, log_prob, sentence


def greedy_decode(sentence, model=None, vocab_file=None, vocab_dir=None, verbose=True):
    """
    Translate and optionally print a sentence using greedy decoding.
    
    Greedy decoding always picks the most probable token at each step.
    Fast and deterministic but can miss better overall translations.

    Args:
        sentence (str): Sentence to translate
        model (tl.Serial): The NMT model
        vocab_file (str): Vocabulary filename (default: from utils)
        vocab_dir (str): Path to vocabulary file (default: from utils)
        verbose (bool): Whether to print input/output (default: True)

    Returns:
        str: The translated sentence
    """
    _, _, translated_sentence = sampling_decode(
        sentence, model, temperature=0.0,
        vocab_file=vocab_file, vocab_dir=vocab_dir
    )
    
    if verbose:
        print("English:", sentence)
        print("German:", translated_sentence)
    
    return translated_sentence


# =============================================================================
# Minimum Bayes Risk (MBR) Decoding
# =============================================================================

def generate_samples(sentence, n_samples, model=None, temperature=0.6,
                    vocab_file=None, vocab_dir=None):
    """
    Generate multiple translation samples.
    
    Args:
        sentence (str): Sentence to translate
        n_samples (int): Number of samples to generate
        model (tl.Serial): The NMT model
        temperature (float): Sampling temperature (default: 0.6)
        vocab_file (str): Vocabulary filename (default: from utils)
        vocab_dir (str): Path to vocabulary file (default: from utils)
        
    Returns:
        tuple: (list of token lists, list of log probabilities)
    """
    samples, log_probs = [], []
    
    for _ in range(n_samples):
        sample, logp, _ = sampling_decode(sentence, model, temperature, 
                                         vocab_file=vocab_file, vocab_dir=vocab_dir)
        samples.append(sample)
        log_probs.append(logp)
    
    return samples, log_probs


def jaccard_similarity(candidate, reference):
    """
    Calculate Jaccard similarity between two token lists.
    
    Jaccard similarity = |intersection| / |union|
    Simple but effective measure of token overlap.

    Args:
        candidate (list): Tokenized candidate translation
        reference (list): Tokenized reference translation

    Returns:
        float: Similarity score (0 to 1)
    """
    can_unigram_set = set(candidate)
    ref_unigram_set = set(reference)
    
    joint_elems = can_unigram_set.intersection(ref_unigram_set)
    all_elems = can_unigram_set.union(ref_unigram_set)
    
    overlap = len(joint_elems) / len(all_elems) if len(all_elems) > 0 else 0
    
    return overlap


def rouge1_similarity(system, reference):
    """
    Calculate ROUGE-1 F1 score between two token lists.
    
    ROUGE-1 is widely used in machine translation evaluation. It computes
    precision and recall based on unigram overlap, then combines them into
    an F1 score.

    Args:
        system (list): Tokenized system translation
        reference (list): Tokenized reference translation

    Returns:
        float: ROUGE-1 F1 score
    """
    sys_counter = Counter(system)
    ref_counter = Counter(reference)
    
    overlap = 0
    for token in sys_counter:
        token_count_sys = sys_counter[token]
        token_count_ref = ref_counter[token]
        overlap += min(token_count_sys, token_count_ref)
    
    precision = overlap / sum(sys_counter.values()) if sum(sys_counter.values()) > 0 else 0
    recall = overlap / sum(ref_counter.values()) if sum(ref_counter.values()) > 0 else 0
    
    if precision + recall != 0:
        rouge1_score = 2 * (precision * recall) / (precision + recall)
    else:
        rouge1_score = 0
    
    return rouge1_score


def average_overlap(similarity_fn, samples, *ignore_params):
    """
    Calculate average overlap score for each sample.
    
    Each sample is compared against all other samples, and the average
    similarity score is computed.

    Args:
        similarity_fn (function): Function to compute similarity
        samples (list of lists): Token lists for each sample
        *ignore_params: Additional parameters (ignored)

    Returns:
        dict: Scores for each sample {index: score}
    """
    scores = {}
    
    for index_candidate, candidate in enumerate(samples):
        overlap = 0
        
        for index_sample, sample in enumerate(samples):
            if index_candidate == index_sample:
                continue
            
            sample_overlap = similarity_fn(candidate, sample)
            overlap += sample_overlap
        
        score = overlap / (len(samples) - 1) if len(samples) > 1 else 0
        scores[index_candidate] = score
    
    return scores


def weighted_avg_overlap(similarity_fn, samples, log_probs):
    """
    Calculate weighted average overlap score for each sample.
    
    Similar to average_overlap, but weights each comparison by the
    log probability of the reference sample.

    Args:
        similarity_fn (function): Function to compute similarity
        samples (list of lists): Token lists for each sample
        log_probs (list): Log probabilities for each sample

    Returns:
        dict: Scores for each sample {index: score}
    """
    scores = {}
    
    for index_candidate, candidate in enumerate(samples):
        overlap, weight_sum = 0.0, 0.0
        
        for index_sample, (sample, logp) in enumerate(zip(samples, log_probs)):
            if index_candidate == index_sample:
                continue
            
            sample_p = float(np.exp(logp))
            weight_sum += sample_p
            
            sample_overlap = similarity_fn(candidate, sample)
            overlap += sample_p * sample_overlap
        
        score = overlap / weight_sum if weight_sum > 0 else 0
        scores[index_candidate] = score
    
    return scores


def mbr_decode(sentence, n_samples, score_fn, similarity_fn, 
               model=None, temperature=0.6, vocab_file=None, vocab_dir=None):
    """
    Translate using Minimum Bayes Risk decoding.
    
    MBR decoding process:
    1. Generate n_samples candidate translations
    2. Score each candidate against all others using similarity_fn
    3. Select the candidate with the highest consensus score

    Args:
        sentence (str): Sentence to translate
        n_samples (int): Number of candidate translations to generate
        score_fn (function): Function to compute sample scores
        similarity_fn (function): Function to compute pairwise similarity
        model (tl.Serial): The NMT model
        temperature (float): Sampling temperature (default: 0.6)
        vocab_file (str): Vocabulary filename (default: from utils)
        vocab_dir (str): Path to vocabulary file (default: from utils)

    Returns:
        tuple: (translated_sentence, best_sample_index, all_scores)
    """
    # Generate candidate translations
    samples, log_probs = generate_samples(sentence, n_samples, model, 
                                         temperature, vocab_file, vocab_dir)
    
    # Score each candidate
    scores = score_fn(similarity_fn, samples, log_probs)
    
    # Select best candidate
    max_score_key = max(scores, key=scores.get)
    
    # Detokenize the best translation
    translated_sentence = detokenize(samples[max_score_key], vocab_file, vocab_dir)
    
    return (translated_sentence, max_score_key, scores)


# =============================================================================
# Convenience Functions
# =============================================================================

def load_model(model_path="model.pkl.gz", **model_kwargs):
    """
    Load a trained NMT model from file.
    
    Args:
        model_path (str): Path to saved model weights
        **model_kwargs: Additional arguments for NMTAttn
        
    Returns:
        tl.Serial: Loaded model ready for inference
    """
    model = NMTAttn(mode='eval', **model_kwargs)
    model.init_from_file(model_path, weights_only=True)
    model = tl.Accelerate(model)
    return model


def translate_sentence(sentence, model_path="model.pkl.gz", temperature=0.0):
    """
    Translate a single sentence (convenience function).
    
    Args:
        sentence (str): English sentence to translate
        model_path (str): Path to saved model weights
        temperature (float): Sampling temperature (0.0 for greedy)
        
    Returns:
        str: German translation
    """
    model = load_model(model_path)
    _, _, translation = sampling_decode(sentence, model, temperature)
    return translation


def mbr_translate(sentence, model_path="model.pkl.gz", n_samples=10):
    """
    Translate using MBR decoding (convenience function).
    
    Args:
        sentence (str): English sentence to translate
        model_path (str): Path to saved model weights
        n_samples (int): Number of candidates for MBR
        
    Returns:
        str: German translation
    """
    model = load_model(model_path)
    translation, _, _ = mbr_decode(
        sentence, n_samples, average_overlap, rouge1_similarity,
        model, temperature=0.6
    )
    return translation

