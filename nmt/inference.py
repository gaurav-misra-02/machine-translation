"""
Inference and decoding strategies for NMT model.

This module implements multiple decoding approaches:
- Greedy decoding (fast, deterministic)
- Sampling decoding (diverse, controllable randomness)
- Minimum Bayes Risk (MBR) decoding (high quality, slower)
"""

from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
from collections import Counter
from trax import layers as tl

from .utils import tokenize, detokenize, EOS, VOCAB_FILE, VOCAB_DIR
from .model import NMTAttn

# Inference default constants
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MBR_TEMPERATURE = 0.6
DEFAULT_N_SAMPLES = 10
DEFAULT_MODEL_PATH = "model.pkl.gz"


def next_symbol(model: tl.Serial, 
                input_tokens: np.ndarray, 
                cur_output_tokens: List[int], 
                temperature: float) -> Tuple[int, float]:
    """
    Predict the next token in the translation.
    
    Handles the autoregressive generation process. Pads the current output
    to the next power of 2 for efficiency and gets the model's prediction.

    Args:
        model: The NMT model
        input_tokens: Tokenized input sentence (1 x n_tokens)
        cur_output_tokens: Previously generated tokens
        temperature: Sampling temperature (0.0 = greedy, 1.0 = random)

    Returns:
        (next_token_index, log_probability)
    
    Raises:
        ValueError: If temperature is negative
    """
    if temperature < 0:
        raise ValueError(f"temperature must be non-negative, got {temperature}")
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


def sampling_decode(input_sentence: str, 
                   model: Optional[tl.Serial] = None, 
                   temperature: float = DEFAULT_TEMPERATURE, 
                   vocab_file: Optional[str] = None, 
                   vocab_dir: Optional[str] = None) -> Tuple[List[int], float, str]:
    """
    Translate a sentence using sampling-based decoding.
    
    Temperature controls the randomness:
    - 0.0: Greedy decoding (always pick most probable)
    - 1.0: Sample from full distribution (most random)
    - 0.3-0.7: Sweet spot for diverse but reasonable translations

    Args:
        input_sentence: Sentence to translate
        model: The NMT model (required)
        temperature: Sampling temperature
        vocab_file: Vocabulary filename (default: from utils)
        vocab_dir: Path to vocabulary file (default: from utils)

    Returns:
        (token_list, log_probability, translated_sentence)
    
    Raises:
        ValueError: If model is None or input_sentence is empty
    """
    if model is None:
        raise ValueError("model cannot be None")
    if not input_sentence or not input_sentence.strip():
        raise ValueError("input_sentence cannot be empty")
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


def greedy_decode(sentence: str, 
                  model: Optional[tl.Serial] = None, 
                  vocab_file: Optional[str] = None, 
                  vocab_dir: Optional[str] = None, 
                  verbose: bool = True) -> str:
    """
    Translate and optionally print a sentence using greedy decoding.
    
    Greedy decoding always picks the most probable token at each step.
    Fast and deterministic but can miss better overall translations.

    Args:
        sentence: Sentence to translate
        model: The NMT model (required)
        vocab_file: Vocabulary filename (default: from utils)
        vocab_dir: Path to vocabulary file (default: from utils)
        verbose: Whether to print input/output

    Returns:
        The translated sentence
    
    Raises:
        ValueError: If model is None or sentence is empty
    """
    if model is None:
        raise ValueError("model cannot be None")
    if not sentence or not sentence.strip():
        raise ValueError("sentence cannot be empty")
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

def generate_samples(sentence: str, 
                    n_samples: int, 
                    model: Optional[tl.Serial] = None, 
                    temperature: float = DEFAULT_MBR_TEMPERATURE,
                    vocab_file: Optional[str] = None, 
                    vocab_dir: Optional[str] = None) -> Tuple[List[List[int]], List[float]]:
    """
    Generate multiple translation samples.
    
    Args:
        sentence: Sentence to translate
        n_samples: Number of samples to generate (must be positive)
        model: The NMT model (required)
        temperature: Sampling temperature
        vocab_file: Vocabulary filename (default: from utils)
        vocab_dir: Path to vocabulary file (default: from utils)
        
    Returns:
        (list of token lists, list of log probabilities)
    
    Raises:
        ValueError: If model is None, n_samples is not positive, or sentence is empty
    """
    if model is None:
        raise ValueError("model cannot be None")
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if not sentence or not sentence.strip():
        raise ValueError("sentence cannot be empty")
    samples, log_probs = [], []
    
    for _ in range(n_samples):
        sample, logp, _ = sampling_decode(sentence, model, temperature, 
                                         vocab_file=vocab_file, vocab_dir=vocab_dir)
        samples.append(sample)
        log_probs.append(logp)
    
    return samples, log_probs


def jaccard_similarity(candidate: List[int], reference: List[int]) -> float:
    """
    Calculate Jaccard similarity between two token lists.
    
    Jaccard similarity = |intersection| / |union|
    Simple but effective measure of token overlap.

    Args:
        candidate: Tokenized candidate translation
        reference: Tokenized reference translation

    Returns:
        Similarity score (0 to 1)
    """
    can_unigram_set = set(candidate)
    ref_unigram_set = set(reference)
    
    joint_elems = can_unigram_set.intersection(ref_unigram_set)
    all_elems = can_unigram_set.union(ref_unigram_set)
    
    overlap = len(joint_elems) / len(all_elems) if len(all_elems) > 0 else 0
    
    return overlap


def rouge1_similarity(system: List[int], reference: List[int]) -> float:
    """
    Calculate ROUGE-1 F1 score between two token lists.
    
    ROUGE-1 is widely used in machine translation evaluation. It computes
    precision and recall based on unigram overlap, then combines them into
    an F1 score.

    Args:
        system: Tokenized system translation
        reference: Tokenized reference translation

    Returns:
        ROUGE-1 F1 score (0 to 1)
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


def average_overlap(similarity_fn: Callable[[List[int], List[int]], float], 
                   samples: List[List[int]], 
                   *ignore_params) -> Dict[int, float]:
    """
    Calculate average overlap score for each sample.
    
    Each sample is compared against all other samples, and the average
    similarity score is computed.

    Args:
        similarity_fn: Function to compute similarity
        samples: Token lists for each sample
        *ignore_params: Additional parameters (ignored)

    Returns:
        Scores for each sample {index: score}
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


def weighted_avg_overlap(similarity_fn: Callable[[List[int], List[int]], float], 
                        samples: List[List[int]], 
                        log_probs: List[float]) -> Dict[int, float]:
    """
    Calculate weighted average overlap score for each sample.
    
    Similar to average_overlap, but weights each comparison by the
    log probability of the reference sample.

    Args:
        similarity_fn: Function to compute similarity
        samples: Token lists for each sample
        log_probs: Log probabilities for each sample

    Returns:
        Scores for each sample {index: score}
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


def mbr_decode(sentence: str, 
               n_samples: int, 
               score_fn: Callable, 
               similarity_fn: Callable[[List[int], List[int]], float], 
               model: Optional[tl.Serial] = None, 
               temperature: float = DEFAULT_MBR_TEMPERATURE, 
               vocab_file: Optional[str] = None, 
               vocab_dir: Optional[str] = None) -> Tuple[str, int, Dict[int, float]]:
    """
    Translate using Minimum Bayes Risk decoding.
    
    MBR decoding process:
    1. Generate n_samples candidate translations
    2. Score each candidate against all others using similarity_fn
    3. Select the candidate with the highest consensus score

    Args:
        sentence: Sentence to translate
        n_samples: Number of candidate translations to generate (must be positive)
        score_fn: Function to compute sample scores
        similarity_fn: Function to compute pairwise similarity
        model: The NMT model (required)
        temperature: Sampling temperature
        vocab_file: Vocabulary filename (default: from utils)
        vocab_dir: Path to vocabulary file (default: from utils)

    Returns:
        (translated_sentence, best_sample_index, all_scores)
    
    Raises:
        ValueError: If model is None, n_samples is not positive, or sentence is empty
    """
    if model is None:
        raise ValueError("model cannot be None")
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if not sentence or not sentence.strip():
        raise ValueError("sentence cannot be empty")
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

def load_model(model_path: str = DEFAULT_MODEL_PATH, **model_kwargs) -> tl.Serial:
    """
    Load a trained NMT model from file.
    
    Args:
        model_path: Path to saved model weights
        **model_kwargs: Additional arguments for NMTAttn
        
    Returns:
        Loaded model ready for inference
    
    Raises:
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If model cannot be loaded
    """
    try:
        model = NMTAttn(mode='eval', **model_kwargs)
        model.init_from_file(model_path, weights_only=True)
        model = tl.Accelerate(model)
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}") from e


def translate_sentence(sentence: str, 
                      model_path: str = DEFAULT_MODEL_PATH, 
                      temperature: float = DEFAULT_TEMPERATURE) -> str:
    """
    Translate a single sentence (convenience function).
    
    Args:
        sentence: English sentence to translate
        model_path: Path to saved model weights
        temperature: Sampling temperature (0.0 for greedy)
        
    Returns:
        German translation
    
    Raises:
        ValueError: If sentence is empty
        FileNotFoundError: If model file doesn't exist
    """
    if not sentence or not sentence.strip():
        raise ValueError("sentence cannot be empty")
    model = load_model(model_path)
    _, _, translation = sampling_decode(sentence, model, temperature)
    return translation


def mbr_translate(sentence: str, 
                  model_path: str = DEFAULT_MODEL_PATH, 
                  n_samples: int = DEFAULT_N_SAMPLES) -> str:
    """
    Translate using MBR decoding (convenience function).
    
    Args:
        sentence: English sentence to translate
        model_path: Path to saved model weights
        n_samples: Number of candidates for MBR (must be positive)
        
    Returns:
        German translation
    
    Raises:
        ValueError: If sentence is empty or n_samples is not positive
        FileNotFoundError: If model file doesn't exist
    """
    if not sentence or not sentence.strip():
        raise ValueError("sentence cannot be empty")
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    model = load_model(model_path)
    translation, _, _ = mbr_decode(
        sentence, n_samples, average_overlap, rouge1_similarity,
        model, temperature=0.6
    )
    return translation

