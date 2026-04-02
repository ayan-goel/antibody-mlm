"""Tokenizer loading and helpers for antibody sequences.

Uses AntiBERTa2's tokenizer (per-amino-acid vocabulary with 28 tokens).
Sequences must be space-separated before tokenization (e.g., "E V Q L V E S").
"""

from transformers import AutoTokenizer, PreTrainedTokenizerBase

ANTIBERTA2_MODEL_NAME = "alchemab/antiberta2"


def load_tokenizer(model_name: str = ANTIBERTA2_MODEL_NAME) -> PreTrainedTokenizerBase:
    """Load the AntiBERTa2 tokenizer from HuggingFace.

    Returns a tokenizer with special tokens [CLS], [SEP], [MASK], [PAD]
    and a per-amino-acid vocabulary (vocab_size=28).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def tokenize_sequence(sequence: str, tokenizer: PreTrainedTokenizerBase) -> str:
    """Convert a raw amino acid string into space-separated format for the tokenizer."""
    return " ".join(list(sequence))
