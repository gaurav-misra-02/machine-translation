# Neural Machine Translation with Attention

A complete implementation of an English-to-German neural machine translation system using LSTM networks with attention mechanism. This project demonstrates the power of attention in handling variable-length sequences and improving translation quality for longer sentences.

## Project Overview

Machine translation is a fundamental task in natural language processing that has numerous real-world applications. This project implements a sequence-to-sequence model with an attention mechanism to translate English sentences to German. The attention layer allows the decoder to focus on relevant parts of the input sentence at each decoding step, significantly improving translation quality compared to vanilla encoder-decoder architectures.

### Key Features

- **Attention Mechanism**: Implements Scaled Dot Product Attention for better context handling
- **Multiple Decoding Strategies**: 
  - Greedy decoding for deterministic translations
  - Temperature-based sampling for diverse outputs
  - Minimum Bayes Risk (MBR) decoding for quality optimization
- **Efficient Training**: Utilizes bucketing to batch sentences of similar lengths
- **Subword Tokenization**: Handles out-of-vocabulary words gracefully using BPE-based subwords

## Architecture

The model consists of three main components:

1. **Encoder**: Multi-layer LSTM that processes the input sentence and generates context vectors
2. **Attention Layer**: Computes alignment scores between encoder and decoder states using scaled dot-product attention
3. **Decoder**: Multi-layer LSTM that generates the translation token by token, guided by attention-weighted context

```
Input (English) → Encoder LSTM → Keys & Values
                                      ↓
Target (German) → Pre-Attention Decoder → Queries → Attention Layer → Context
                                                            ↓
                                                      Decoder LSTM → Output (German)
```

## Installation

### Prerequisites

- Python 3.7+
- CUDA-compatible GPU (optional, but recommended for faster training)

### Setup

1. Clone this repository:
```bash
cd machine-translation
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the training data:
The project uses the OPUS medical translation dataset, which will be automatically downloaded on first run.

## Usage

### Quick Start with Command-Line Scripts

The easiest way to use the model is through the provided scripts:

**Train a model:**
```bash
python scripts/train.py --steps 1000 --output_dir checkpoints/
```

**Translate sentences:**
```bash
# Greedy decoding (fast, deterministic)
python scripts/translate.py --sentence "I love machine learning" --method greedy

# Sampling with temperature (diverse outputs)
python scripts/translate.py --sentence "Hello, how are you?" --method sampling --temperature 0.7

# MBR decoding (highest quality)
python scripts/translate.py --sentence "This is complex" --method mbr --samples 10
```

### Using as a Python Package

Import and use the modular components:

```python
from nmt import get_data_pipeline, train_model, NMTAttn

# Load data
train_stream, eval_stream = get_data_pipeline(max_length=512)

# Train model
model_config = {
    'd_model': 1024,
    'n_encoder_layers': 2,
    'n_decoder_layers': 2,
    'n_attention_heads': 4
}

training_loop = train_model(
    train_stream, 
    eval_stream, 
    n_steps=100,
    model_config=model_config
)
```

### Inference with Pre-trained Model

```python
from nmt import translate_sentence, mbr_translate

# Simple translation using greedy decoding
translation = translate_sentence("I love machine learning.")
print(translation)  # Output: "Ich liebe maschinelles Lernen."

# High-quality translation with MBR decoding
translation = mbr_translate("This is a complex sentence.", n_samples=10)
print(translation)
```

### Advanced Usage

For more control over the inference process:

```python
from nmt import load_model, sampling_decode, mbr_decode, rouge1_similarity, average_overlap

# Load model
model = load_model("model.pkl.gz")

# Greedy decoding
tokens, log_prob, translation = sampling_decode(
    "Hello world", 
    model, 
    temperature=0.0
)

# MBR decoding with custom similarity function
translation, best_idx, scores = mbr_decode(
    "Hello world",
    n_samples=20,
    score_fn=average_overlap,
    similarity_fn=rouge1_similarity,
    model=model,
    temperature=0.6
)
```

### Interactive Jupyter Notebook

For an interactive experience with visualizations:

```bash
jupyter notebook Neural_Machine_Translation.ipynb
```

## Model Performance

The model achieves solid translation quality on medical domain text:

- **Training Loss**: ~4.2 after convergence
- **Validation Accuracy**: ~45% token-level accuracy
- **Translation Quality**: Produces fluent German translations for most common English sentences

### Example Translations

| English | German Translation | Method |
|---------|-------------------|--------|
| "I am hungry" | "Ich bin hungrig" | Greedy |
| "She speaks English and German." | "Sie spricht Englisch und Deutsch." | MBR |
| "How are you today?" | "Wie geht es Ihnen heute?" | Greedy |

## Decoding Strategies

### Greedy Decoding
Always selects the most probable token at each step. Fast but can miss better overall translations.

```python
translation = greedy_decode("Hello world", model)
```

### Temperature Sampling
Introduces randomness controlled by temperature (0.0 = greedy, 1.0 = fully random).

```python
translation = sampling_decode("Hello world", model, temperature=0.6)
```

### Minimum Bayes Risk (MBR) Decoding
Generates multiple candidate translations and selects the one with highest consensus score. Slower but produces higher quality translations.

```python
translation = mbr_decode("Hello world", model, n_samples=10, similarity_fn=rouge1_similarity)
```

## Project Structure

```
machine-translation/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore patterns
│
├── nmt/                               # Main package
│   ├── __init__.py                    # Package initialization
│   ├── data.py                        # Data loading and preprocessing
│   ├── model.py                       # Model architecture
│   ├── training.py                    # Training configuration
│   ├── inference.py                   # Decoding strategies
│   └── utils.py                       # Helper functions
│
├── scripts/                           # Command-line scripts
│   ├── train.py                       # Training script
│   └── translate.py                   # Translation script
│
├── Neural_Machine_Translation.py      # Legacy monolithic module
├── Neural_Machine_Translation.ipynb   # Interactive notebook
│
├── model.pkl.gz                       # Pre-trained model weights
├── data/                              # Training data (auto-downloaded)
│   └── ende_32k.subword               # Vocabulary file
└── output_dir/                        # Training checkpoints
```

## Module Overview

The codebase is organized into clean, modular components:

### Core Package (`nmt/`)

- **`data.py`**: Data loading and preprocessing pipeline
  - Load OPUS medical corpus
  - Tokenization with BPE subwords
  - Bucketing for efficient batching
  - Masking for padding tokens

- **`model.py`**: Neural network architecture
  - Input encoder (LSTM stack)
  - Pre-attention decoder
  - Attention mechanism setup
  - Complete NMTAttn model

- **`training.py`**: Training configuration
  - Training task setup
  - Evaluation task setup
  - Learning rate scheduling
  - Checkpoint management

- **`inference.py`**: Decoding strategies
  - Greedy decoding
  - Sampling with temperature
  - MBR (Minimum Bayes Risk) decoding
  - ROUGE and Jaccard similarity metrics

- **`utils.py`**: Helper utilities
  - Tokenize/detokenize functions
  - EOS token handling
  - Vocabulary configuration

### Scripts (`scripts/`)

- **`train.py`**: Command-line training script with argparse
- **`translate.py`**: Command-line translation script with multiple decoding options

### Legacy Files

- **`Neural_Machine_Translation.py`**: Original monolithic module (kept for compatibility)
- **`Neural_Machine_Translation.ipynb`**: Interactive Jupyter notebook

## Technical Details

### Attention Mechanism

The model uses Scaled Dot-Product Attention:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- Q (Queries): Decoder hidden states
- K (Keys): Encoder hidden states
- V (Values): Encoder hidden states
- d_k: Dimension of the keys (used for scaling)

### Training Configuration

- **Optimizer**: Adam with learning rate warmup
- **Learning Rate Schedule**: Warmup for 1000 steps, then inverse square root decay
- **Batch Size**: Dynamic (bucketed by sentence length)
- **Loss Function**: Cross-entropy loss
- **Gradient Clipping**: Applied to prevent exploding gradients

### Data Preprocessing

1. **Tokenization**: Subword tokenization using BPE (Byte Pair Encoding)
2. **Bucketing**: Sentences grouped by length for efficient batching
3. **Padding**: Sequences padded to nearest power of 2
4. **EOS Token**: End-of-sequence token appended to all sentences

## Challenges and Learnings

Building this project involved several interesting challenges:

1. **Memory Management**: Training on long sequences requires careful memory management through bucketing
2. **Attention Implementation**: Properly implementing multi-headed attention with correct tensor shapes
3. **Decoding Trade-offs**: Balancing translation quality with inference speed across different decoding methods
4. **Vanishing Gradients**: Using residual connections and proper initialization to maintain gradient flow

## Future Improvements

- [ ] Implement Transformer architecture for comparison
- [ ] Add BLEU score evaluation
- [ ] Support for additional language pairs
- [ ] Beam search decoding
- [ ] Fine-tuning on specific domains
- [ ] Web interface for interactive translation

## Code Quality and Best Practices

This codebase follows modern Python best practices and professional coding standards:

### Type Safety
- **Complete Type Hints**: All functions include type hints for parameters and return values
- **IDE Support**: Enhanced autocomplete and IntelliSense
- **Early Error Detection**: Catch type-related bugs during development

```python
def tokenize(input_str: str, 
              vocab_file: Optional[str] = None, 
              vocab_dir: Optional[str] = None) -> np.ndarray:
    """Type-safe tokenization function."""
    pass
```

### Input Validation
- **Comprehensive Validation**: All functions validate inputs with descriptive error messages
- **Early Failure**: Fail fast with clear explanations
- **Parameter Constraints**: Check ranges, non-empty values, and valid options

```python
def train_model(n_steps: int, ...) -> training.Loop:
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    # ...
```

### Configuration Management
- **Type-Safe Configs**: Dataclasses for all configuration objects
- **Built-in Validation**: Automatic parameter validation
- **Clear Documentation**: Self-documenting configuration classes

```python
from nmt import ModelConfig

config = ModelConfig(
    d_model=512,
    n_encoder_layers=3,
    n_attention_heads=8
)
# Automatically validates all parameters
```

Available configuration classes:
- `ModelConfig` - Model architecture parameters
- `DataConfig` - Data loading and preprocessing
- `TrainingConfig` - Training hyperparameters
- `InferenceConfig` - Inference settings

### Logging and Monitoring
- **Structured Logging**: Professional logging system with timestamps
- **Log Levels**: INFO, DEBUG, WARNING, ERROR support
- **No Print Statements**: All output uses proper logging

```python
import logging
logger = logging.getLogger(__name__)

logger.info("Training started")
logger.error(f"Failed to load model: {error}")
```

### Error Handling
- **Graceful Failures**: Try-catch blocks with informative messages
- **Resource Cleanup**: Proper handling of KeyboardInterrupt
- **Exit Codes**: Appropriate exit codes for automation
- **Helpful Suggestions**: Error messages include potential solutions

### Named Constants
All magic numbers extracted to named constants for maintainability:

```python
# Data pipeline
DEFAULT_MAX_LENGTH = 512
DEFAULT_BOUNDARIES = [8, 16, 32, 64, 128, 256, 512]
PADDING_ID = 0

# Model architecture
DEFAULT_VOCAB_SIZE = 33300
DEFAULT_D_MODEL = 1024
DEFAULT_N_ATTENTION_HEADS = 4

# Training
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_WARMUP_STEPS = 1000
```

### Documentation Standards
- **Comprehensive Docstrings**: All functions include detailed documentation
- **Parameter Descriptions**: Clear explanation of all parameters
- **Return Values**: Documented return types and meanings
- **Examples**: Usage examples in docstrings
- **Raises Sections**: All possible exceptions documented

### Code Organization
- **Modular Structure**: Clean separation of concerns (see [ARCHITECTURE.md](ARCHITECTURE.md))
- **Single Responsibility**: Each module has a focused purpose
- **Consistent Naming**: `snake_case` for functions, `PascalCase` for classes
- **Import Organization**: Standard library, third-party, then local imports

### Quality Metrics
- ✅ **Zero Linter Errors**: Clean code that passes all linting checks
- ✅ **Type Hints Coverage**: 100% of functions have complete type hints
- ✅ **Input Validation**: All user-facing functions validate inputs
- ✅ **Professional Logging**: Structured logging throughout
- ✅ **Error Messages**: Clear, actionable error messages

### Using Configuration Classes

Instead of passing many parameters:

```python
# Before
model = NMTAttn(
    input_vocab_size=33300,
    target_vocab_size=33300,
    d_model=1024,
    n_encoder_layers=2,
    n_decoder_layers=2,
    n_attention_heads=4,
    attention_dropout=0.0,
    mode='train'
)

# After - cleaner and type-safe
from nmt import ModelConfig, NMTAttn

config = ModelConfig(
    d_model=1024,
    n_encoder_layers=2,
    n_decoder_layers=2,
    n_attention_heads=4
)
model = NMTAttn(**config.to_dict())
```

### Benefits
1. **Maintainability**: Easy to understand and modify
2. **Reliability**: Input validation prevents many bugs
3. **Debuggability**: Structured logging and clear error messages
4. **Scalability**: Modular structure supports growth
5. **Production-Ready**: Professional code quality standards

## Dependencies

Key libraries used:
- **Trax**: Google's deep learning library for seq2seq models
- **NumPy**: Numerical computations
- **TensorFlow**: Backend for data loading
- **Termcolor**: Colored terminal output

See `requirements.txt` for complete list with versions.

## References

- Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural Machine Translation by Jointly Learning to Align and Translate.
- Vaswani, A., et al. (2017). Attention Is All You Need.
- OPUS Corpus: http://opus.nlpl.eu/

## License

This project is available for educational and research purposes.

## Acknowledgments

This project was built using the Trax library by Google Brain team and trained on the OPUS medical translation corpus.

---

**Note**: This is a research/educational project. For production translation systems, consider using established frameworks like Hugging Face Transformers or Google Translate API.

