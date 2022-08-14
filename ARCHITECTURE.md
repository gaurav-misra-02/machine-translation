# Architecture Overview

This document explains the modular architecture of the Neural Machine Translation project.

## Why Modular?

The original `Neural_Machine_Translation.py` was a single 850+ line file. While functional, this made it:
- Hard to navigate
- Difficult to test individual components
- Challenging to reuse parts in other projects
- Less maintainable as the codebase grows

The new modular structure addresses all these issues while maintaining full backward compatibility.

## New Structure

```
machine-translation/
├── nmt/                    # Main Python package
│   ├── __init__.py        # Exposes public API
│   ├── data.py            # ~110 lines - Data pipeline
│   ├── model.py           # ~170 lines - Architecture
│   ├── training.py        # ~100 lines - Training config
│   ├── inference.py       # ~340 lines - All decoding
│   └── utils.py           # ~90 lines - Helpers
│
└── scripts/               # Command-line tools
    ├── train.py           # ~120 lines - Training CLI
    └── translate.py       # ~130 lines - Translation CLI
```

**Total: ~1,060 lines across 7 focused modules vs. 1 monolithic 850-line file**

## Module Responsibilities

### `nmt/data.py` - Data Pipeline
**Purpose**: Everything related to loading and preprocessing data

**Key Functions**:
- `load_data()` - Load OPUS corpus
- `prepare_data_pipeline()` - Complete preprocessing
- `get_data_pipeline()` - Convenience wrapper

**Why separate**: Data loading is independent of model architecture. You could swap datasets without touching model code.

### `nmt/model.py` - Neural Network Architecture  
**Purpose**: All model architecture components

**Key Functions**:
- `input_encoder_fn()` - Encoder LSTM stack
- `pre_attention_decoder_fn()` - Decoder before attention
- `prepare_attention_input()` - Q, K, V, mask setup
- `NMTAttn()` - Complete model

**Why separate**: Clean separation between model architecture and training/inference. Easy to experiment with different architectures.

### `nmt/training.py` - Training Configuration
**Purpose**: Everything needed to train the model

**Key Functions**:
- `create_train_task()` - Training task setup
- `create_eval_task()` - Evaluation task setup  
- `train_model()` - Complete training loop
- `train_from_config()` - Config-based training

**Why separate**: Training logic separate from model definition. Can easily modify training parameters without touching architecture.

### `nmt/inference.py` - Decoding Strategies
**Purpose**: All inference and translation logic

**Key Functions**:
- `next_symbol()` - Predict next token
- `sampling_decode()` - Sample-based decoding
- `greedy_decode()` - Greedy decoding
- `mbr_decode()` - MBR decoding
- `generate_samples()` - Generate candidates
- `jaccard_similarity()` - Similarity metric
- `rouge1_similarity()` - ROUGE-1 metric
- `average_overlap()` - Scoring function
- `weighted_avg_overlap()` - Weighted scoring
- `load_model()` - Load trained model
- `translate_sentence()` - Simple translation
- `mbr_translate()` - MBR translation

**Why separate**: Largest module because it contains all decoding strategies. Each strategy is independent and can be tested separately.

### `nmt/utils.py` - Helper Functions
**Purpose**: Shared utilities used across modules

**Key Functions**:
- `tokenize()` - Text to tokens
- `detokenize()` - Tokens to text
- `append_eos()` - Add end-of-sentence markers

**Global Constants**:
- `VOCAB_FILE` - Vocabulary filename
- `VOCAB_DIR` - Vocabulary directory
- `EOS` - End-of-sentence token

**Why separate**: Prevents code duplication. These utilities are used by multiple modules.

### `nmt/__init__.py` - Public API
**Purpose**: Define what users can import

**Exports**:
- All major functions from each module
- Clean namespace for package users
- Version information

**Why needed**: Makes `from nmt import translate_sentence` work cleanly.

## Command-Line Scripts

### `scripts/train.py`
**Purpose**: Train models from the command line

**Features**:
- Argument parsing for all training parameters
- Progress display
- Error handling
- Checkpoint management

**Usage**:
```bash
python scripts/train.py --steps 1000 --d_model 512
```

### `scripts/translate.py`
**Purpose**: Translate sentences from the command line

**Features**:
- Multiple decoding methods (greedy, sampling, MBR)
- Configurable temperature and samples
- Nice output formatting

**Usage**:
```bash
python scripts/translate.py --sentence "Hello" --method mbr
```

## Usage Patterns

### As a Package (Recommended)
```python
from nmt import get_data_pipeline, train_model
from nmt import translate_sentence, mbr_translate

# Use modular components
train_stream, eval_stream = get_data_pipeline()
```

### Command Line (Easiest)
```bash
python scripts/train.py --steps 100
python scripts/translate.py --sentence "Hello world"
```

### Legacy Compatibility
```python
# Old monolithic file still works
from Neural_Machine_Translation import translate_sentence
```

## Benefits of New Structure

### 1. **Easier to Understand**
Each file is ~100-340 lines and has a single responsibility. Much easier to read than 850 lines.

### 2. **Better Testing**
Can test each module independently:
```python
from nmt.model import input_encoder_fn
# Test just the encoder
```

### 3. **Code Reuse**
Easy to use parts in other projects:
```python
from nmt.inference import rouge1_similarity
# Use ROUGE-1 in a different project
```

### 4. **Maintainability**
Bug fixes and improvements are isolated:
- Bug in decoding? → Check `inference.py`
- Data issue? → Check `data.py`
- Training problem? → Check `training.py`

### 5. **Extensibility**
Easy to add new features:
- New decoding strategy? → Add to `inference.py`
- New architecture? → Add to `model.py`
- New training technique? → Add to `training.py`

### 6. **Professional Structure**
Follows Python best practices:
- Package structure (`nmt/`)
- Clean imports
- Separation of concerns
- Command-line tools

## Migration from Monolithic File

If you have existing code using `Neural_Machine_Translation.py`:

### Option 1: Keep Using It (Backward Compatible)
```python
# Still works!
from Neural_Machine_Translation import translate_sentence
```

### Option 2: Migrate to Modular (Recommended)
```python
# Old
from Neural_Machine_Translation import translate_sentence

# New (same interface!)
from nmt import translate_sentence
```

### Option 3: Use Specific Modules
```python
# Import only what you need
from nmt.inference import greedy_decode, mbr_decode
from nmt.model import NMTAttn
from nmt.data import get_data_pipeline
```

## Testing Individual Components

The modular structure makes testing much easier:

```python
# Test data pipeline
from nmt.data import prepare_data_pipeline
# Run tests on data preprocessing

# Test model components
from nmt.model import input_encoder_fn
# Test encoder independently

# Test inference
from nmt.inference import rouge1_similarity
# Test similarity metrics
```

## Adding New Features

### Example: Adding a New Decoding Strategy

1. Open `nmt/inference.py`
2. Add your function:
```python
def beam_search_decode(sentence, model, beam_width=5):
    # Implementation here
    pass
```
3. Add to `nmt/__init__.py`:
```python
from .inference import beam_search_decode
__all__ = [..., 'beam_search_decode']
```
4. Done! Now available as: `from nmt import beam_search_decode`

## Conclusion

The modular structure provides:
- ✅ Better organization
- ✅ Easier maintenance  
- ✅ Improved testability
- ✅ Professional code structure
- ✅ Backward compatibility

All without changing the functionality or API!

