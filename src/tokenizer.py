"""
BPE Tokenizer for the EN→DE NMT pipeline.

Trains a shared Byte Pair Encoding (BPE) vocabulary over the combined
English + German corpus from Multi30k. A joint vocabulary is required for
weight tying in Seq2SeqTransformer (src/tgt embedding matrices are shared).

Workflow:
    1.  Call `train_tokenizer(corpus_iterator, save_dir)` once to fit BPE
        and persist the model + vocab files.
    2.  Call `load_tokenizer(save_dir)` at training / inference time to
        recover the fitted tokenizer.
    3.  Use `TokenizerWrapper` for all encode / decode operations — it
        handles special tokens, padding, and batch collation.

Special tokens (fixed indices for mask construction):
    [PAD] = 0
    [UNK] = 1
    [SOS] = 2
    [EOS] = 3
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
from tokenizers.processors import TemplateProcessing
from tokenizers.normalizers import Lowercase


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
SOS_TOKEN = "[SOS]"
EOS_TOKEN = "[EOS]"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]

PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

_TOKENIZER_FILE = "tokenizer.json"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_tokenizer(
    corpus_iterator: Iterator[str],
    save_dir: str | Path,
    vocab_size: int = 16_000,
    min_frequency: int = 2,
) -> "TokenizerWrapper":
    """
    Trains a shared BPE tokenizer from scratch on the provided text iterator.

    The iterator should yield raw sentences from **both** source and target
    languages interleaved so BPE merges are learned jointly.

    Args:
        corpus_iterator: Iterator yielding raw unicode strings (one sentence
                         per item). Caller is responsible for including both
                         EN and DE sentences.
        save_dir:        Directory where `tokenizer.json` will be saved.
        vocab_size:      Target vocabulary size (paper uses 37k for WMT;
                         16k is sufficient for Multi30k ~29k sentences).
        min_frequency:   Minimum pair frequency for a BPE merge to be learned.

    Returns:
        A ready-to-use `TokenizerWrapper` backed by the trained tokenizer.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Model ---
    tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))

    # --- Normalizer: lowercase only ---
    # NFD + StripAccents was removing German umlauts (ä→a, ö→o, ü→u, ß→ss)
    # which hurts translation quality. Lowercase alone preserves them.
    tokenizer.normalizer = Lowercase()

    # --- Pre-tokenizer: Whitespace + punctuation split ---
    # ByteLevel pre-tokenizer would avoid the OOV problem entirely but adds
    # a byte-level prefix (Ġ) that complicates detokenization. Whitespace
    # split + BPE UNK is simpler and sufficient for Multi30k's domain.
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # --- Decoder: merges BPE pieces back with spaces ---
    tokenizer.decoder = decoders.BPEDecoder()

    # --- Trainer ---
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,  # guarantees indices 0-3 are reserved
        show_progress=True,
    )

    # tokenizers library accepts an iterator of strings directly
    tokenizer.train_from_iterator(corpus_iterator, trainer=trainer)

    # --- Post-processor: auto-wrap every sequence with [SOS] … [EOS] ---
    tokenizer.post_processor = TemplateProcessing(
        single=f"{SOS_TOKEN} $A {EOS_TOKEN}",
        pair=f"{SOS_TOKEN} $A {EOS_TOKEN} {SOS_TOKEN} $B {EOS_TOKEN}",
        special_tokens=[
            (SOS_TOKEN, tokenizer.token_to_id(SOS_TOKEN)),
            (EOS_TOKEN, tokenizer.token_to_id(EOS_TOKEN)),
        ],
    )

    # Persist
    out_path = save_dir / _TOKENIZER_FILE
    tokenizer.save(str(out_path))
    print(f"Tokenizer saved -> {out_path}  (vocab size: {tokenizer.get_vocab_size()})")

    return TokenizerWrapper(tokenizer)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_tokenizer(save_dir: str | Path) -> "TokenizerWrapper":
    """
    Loads a previously trained tokenizer from `save_dir/tokenizer.json`.

    Args:
        save_dir: Directory containing `tokenizer.json`.

    Returns:
        A ready-to-use `TokenizerWrapper`.

    Raises:
        FileNotFoundError: If no tokenizer file exists at the expected path.
    """
    path = Path(save_dir) / _TOKENIZER_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"No tokenizer found at '{path}'. "
            "Run `train_tokenizer()` first to fit and save the BPE model."
        )
    tokenizer = Tokenizer.from_file(str(path))
    return TokenizerWrapper(tokenizer)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class TokenizerWrapper:
    """
    Thin wrapper around a HuggingFace `Tokenizer` instance.

    Provides encode / decode / batch-encode utilities with a consistent
    interface used by `dataset.py` and `translate.py`.

    Special token indices are fixed at construction time and exposed as
    attributes so callers never need to look them up by string.

    Attributes:
        pad_idx: int — index of [PAD]
        unk_idx: int — index of [UNK]
        sos_idx: int — index of [SOS]
        eos_idx: int — index of [EOS]
        vocab_size: int — full vocabulary size
    """

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tok = tokenizer
        self._tok.enable_padding(pad_id=PAD_IDX, pad_token=PAD_TOKEN)

        # Verify special token ordering is as expected
        self._assert_special_token_ids()

        self.pad_idx: int = PAD_IDX
        self.unk_idx: int = UNK_IDX
        self.sos_idx: int = SOS_IDX
        self.eos_idx: int = EOS_IDX

    # ------------------------------------------------------------------
    # Core properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def token_to_id(self, token: str) -> int:
        return self._tok.token_to_id(token)

    def id_to_token(self, idx: int) -> str:
        return self._tok.id_to_token(idx)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """
        Encodes a single sentence.

        The post-processor automatically prepends [SOS] and appends [EOS].

        Args:
            text: Raw unicode string.

        Returns:
            List of integer token ids including [SOS] and [EOS].
        """
        return self._tok.encode(text).ids

    def encode_batch(
        self,
        texts: list[str],
        max_length: int | None = None,
    ) -> tuple[list[list[int]], list[list[int]]]:
        """
        Encodes a batch of sentences with dynamic padding to the longest
        sequence in the batch (or `max_length` if specified).

        Args:
            texts:      List of raw strings.
            max_length: Optional hard cap on sequence length. Sequences
                        longer than this are truncated.

        Returns:
            ids:          (N, padded_len) — token id lists
            attention_masks: (N, padded_len) — 1 for real tokens, 0 for PAD
        """
        if max_length is not None:
            self._tok.enable_truncation(max_length=max_length)
        else:
            self._tok.no_truncation()

        # Padding length is set to longest in batch automatically
        self._tok.enable_padding(pad_id=PAD_IDX, pad_token=PAD_TOKEN)

        encodings = self._tok.encode_batch(texts)
        ids = [enc.ids for enc in encodings]
        attention_masks = [enc.attention_mask for enc in encodings]

        return ids, attention_masks

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """
        Decodes a list of token ids back to a string.

        Args:
            ids:                  List of integer token ids.
            skip_special_tokens:  If True, strips [SOS], [EOS], [PAD], [UNK].

        Returns:
            Decoded unicode string.
        """
        # BPEDecoder with Whitespace pre-tokenizer has no end-of-word markers,
        # so it concatenates all tokens without spaces. We join token strings
        # with spaces instead — both hypothesis and reference go through the
        # same path, so BLEU comparison remains fair.
        special_ids = {self.pad_idx, self.unk_idx, self.sos_idx, self.eos_idx}
        tokens = []
        for id in ids:
            if skip_special_tokens and id in special_ids:
                continue
            token = self._tok.id_to_token(id)
            if token is not None:
                tokens.append(token)
        return " ".join(tokens)

    def decode_batch(
        self,
        batch_ids: list[list[int]],
        skip_special_tokens: bool = True,
    ) -> list[str]:
        """
        Decodes a batch of id sequences.

        Args:
            batch_ids:            List of id lists.
            skip_special_tokens:  Strip special tokens from output.

        Returns:
            List of decoded strings.
        """
        return [self.decode(ids, skip_special_tokens) for ids in batch_ids]

    # ------------------------------------------------------------------
    # Internal validation
    # ------------------------------------------------------------------

    def _assert_special_token_ids(self) -> None:
        """
        Verifies that [PAD]=0, [UNK]=1, [SOS]=2, [EOS]=3 after training.
        The BpeTrainer assigns special_tokens in declaration order, but this
        makes the contract explicit and catches misconfiguration early.
        """
        expected = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX,
                    SOS_TOKEN: SOS_IDX, EOS_TOKEN: EOS_IDX}
        for token, expected_id in expected.items():
            actual_id = self._tok.token_to_id(token)
            if actual_id != expected_id:
                raise RuntimeError(
                    f"Special token id mismatch: '{token}' expected {expected_id}, "
                    f"got {actual_id}. Re-train the tokenizer to fix the ordering."
                )

    def __repr__(self) -> str:
        return (
            f"TokenizerWrapper(vocab_size={self.vocab_size}, "
            f"pad={self.pad_idx}, sos={self.sos_idx}, eos={self.eos_idx})"
        )


# ---------------------------------------------------------------------------
# CLI entry point — trains from a flat text file
# ---------------------------------------------------------------------------

def _iter_lines(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a shared BPE tokenizer on a combined EN+DE corpus file."
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Path to a plain-text file with one sentence per line (EN and DE mixed).",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("data/tokenizer"),
        help="Directory to save tokenizer.json (default: data/tokenizer).",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=16_000,
        help="Target BPE vocabulary size (default: 16000).",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum merge frequency (default: 2).",
    )
    args = parser.parse_args()

    wrapper = train_tokenizer(
        corpus_iterator=_iter_lines(args.corpus),
        save_dir=args.save_dir,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
    print(wrapper)

    # Quick smoke-test
    sample_en = "Two young, White males are outside near many bushes."
    sample_de = "Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche."
    for text in (sample_en, sample_de):
        ids = wrapper.encode(text)
        decoded = wrapper.decode(ids)
        print(f"  IN : {text}")
        print(f"  IDS: {ids[:10]} ... (len={len(ids)})")
        print(f"  OUT: {decoded}")
        print()
