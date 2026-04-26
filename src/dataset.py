"""
Data loading and preprocessing for the EN→DE NMT pipeline.

Pipeline overview:
    1. `download_multi30k()`       — fetches Multi30k via HuggingFace `datasets`
    2. `download_extra_dataset()`  — fetches any additional HF parallel dataset
    3. `build_corpus_file()`       — writes combined EN+DE text for BPE training
    4. `TranslationDataset`        — torch Dataset wrapping tokenized (src, tgt) pairs
    5. `collate_fn`                — dynamic padding + mask construction per batch
    6. `build_dataloaders()`       — returns train/val/test DataLoaders

Supported extra datasets (--extra-dataset flag in train.py):
    "opus_books"   → opus_books           de-en  (~51k literary pairs)
    "wmt14"        → wmt14                de-en  (4.5M news pairs, use --extra-max to cap)
    "opus100"      → Helsinki-NLP/opus-100 de-en  (1M diverse pairs)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Must be imported before torch initialises CUDA to avoid a segfault on Windows
# caused by a conflict between the datasets C extensions and the CUDA runtime.
from datasets import load_dataset as hf_load_dataset

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from tokenizer import TokenizerWrapper, load_tokenizer, train_tokenizer, PAD_IDX


# ---------------------------------------------------------------------------
# Step 1 — Download Multi30k
# ---------------------------------------------------------------------------

def download_multi30k(cache_dir: str | Path = "data/raw") -> dict[str, Any]:
    """
    Downloads the Multi30k EN→DE dataset from HuggingFace Hub.

    The `bentrevett/multi30k` dataset card exposes three splits:
        train  — 29,000 sentence pairs
        validation — 1,014 sentence pairs
        test   — 1,000 sentence pairs

    Each example has keys: "en" (str), "de" (str).

    Args:
        cache_dir: Local directory for HuggingFace dataset cache.

    Returns:
        A `DatasetDict` with keys "train", "validation", "test".
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading Multi30k (bentrevett/multi30k) ...")
    dataset = hf_load_dataset("bentrevett/multi30k", cache_dir=str(cache_dir))
    print(
        f"  train={len(dataset['train']):,}  "
        f"val={len(dataset['validation']):,}  "
        f"test={len(dataset['test']):,}"
    )
    return dataset


# ---------------------------------------------------------------------------
# Step 2 — Download extra dataset (optional)
# ---------------------------------------------------------------------------

# Maps short names to (hf_name, hf_config) and the field extractor function.
_EXTRA_DATASETS: dict[str, tuple[str, str | None, callable]] = {
    "opus_books": (
        "opus_books", "de-en",
        lambda ex: (ex["translation"]["en"], ex["translation"]["de"]),
    ),
    "wmt14": (
        "wmt14", "de-en",
        lambda ex: (ex["translation"]["en"], ex["translation"]["de"]),
    ),
    "opus100": (
        "Helsinki-NLP/opus-100", "de-en",
        lambda ex: (ex["translation"]["en"], ex["translation"]["de"]),
    ),
}


def download_extra_dataset(
    name: str,
    max_examples: int | None = None,
    cache_dir: str | Path = "data/raw",
) -> list[dict[str, str]]:
    """
    Downloads an additional parallel dataset and returns a list of
    {"en": str, "de": str} dicts (same format as Multi30k).

    Args:
        name:         Short name — one of "opus_books", "wmt14", "tatoeba".
        max_examples: Cap the number of pairs (useful for large datasets like wmt14).
        cache_dir:    HuggingFace cache directory.

    Returns:
        List of {"en": str, "de": str} dicts from the training split.
    """
    if name not in _EXTRA_DATASETS:
        raise ValueError(
            f"Unknown extra dataset '{name}'. "
            f"Choose from: {list(_EXTRA_DATASETS.keys())}"
        )

    hf_name, hf_config, extractor = _EXTRA_DATASETS[name]
    print(f"Downloading extra dataset: {hf_name} ({hf_config}) ...")

    split = "train" if name != "tatoeba" else "test"  # tatoeba only has test split
    ds = hf_load_dataset(hf_name, hf_config, split=split, cache_dir=str(cache_dir))

    pairs: list[dict[str, str]] = []
    for ex in ds:
        en, de = extractor(ex)
        en, de = en.strip(), de.strip()
        if en and de:
            pairs.append({"en": en, "de": de})
        if max_examples and len(pairs) >= max_examples:
            break

    print(f"  Loaded {len(pairs):,} extra pairs from {hf_name}")
    return pairs


# ---------------------------------------------------------------------------
# Step 3 — Build combined corpus for BPE training
# ---------------------------------------------------------------------------

def build_corpus_file(
    dataset: Any,
    output_path: str | Path = "data/raw/combined_corpus.txt",
    splits: tuple[str, ...] = ("train",),
    extra_pairs: list[dict[str, str]] | None = None,
) -> Path:
    """
    Writes one sentence per line to a plain-text file, interleaving EN and DE
    sentences from the specified splits, plus any extra pairs.

    Args:
        dataset:     HuggingFace DatasetDict returned by `download_multi30k`.
        output_path: Destination file path.
        splits:      Which splits to include. Default: ("train",) only.
        extra_pairs: Additional {"en", "de"} dicts to append.

    Returns:
        Path to the written corpus file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with output_path.open("w", encoding="utf-8") as f:
        for split in splits:
            for example in dataset[split]:
                f.write(example["en"].strip() + "\n")
                f.write(example["de"].strip() + "\n")
                total += 2
        if extra_pairs:
            for pair in extra_pairs:
                f.write(pair["en"].strip() + "\n")
                f.write(pair["de"].strip() + "\n")
                total += 2

    print(f"Corpus written -> {output_path}  ({total:,} lines)")
    return output_path


# ---------------------------------------------------------------------------
# Step 4 — torch Dataset
# ---------------------------------------------------------------------------

class TranslationDataset(Dataset):
    """
    Wraps a list of {"en", "de"} dicts (or HuggingFace split) into a Dataset.

    Accepts:
        examples: HuggingFace Dataset split OR list of {"en": str, "de": str}.
    """

    def __init__(
        self,
        examples: Any,
        tokenizer: TokenizerWrapper,
        max_src_len: int = 128,
        max_tgt_len: int = 128,
    ) -> None:
        print(f"  Tokenizing {len(examples):,} examples ...", end=" ", flush=True)
        self.data: list[tuple[list[int], list[int]]] = []

        for example in examples:
            src_ids = tokenizer.encode(example["en"])
            tgt_ids = tokenizer.encode(example["de"])

            if len(src_ids) > max_src_len:
                src_ids = src_ids[: max_src_len - 1] + [tokenizer.eos_idx]
            if len(tgt_ids) > max_tgt_len:
                tgt_ids = tgt_ids[: max_tgt_len - 1] + [tokenizer.eos_idx]

            self.data.append((src_ids, tgt_ids))

        print("done.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        return self.data[idx]


# ---------------------------------------------------------------------------
# Step 4 — Collate function (dynamic padding + mask construction)
# ---------------------------------------------------------------------------

def collate_fn(
    batch: list[tuple[list[int], list[int]]],
    pad_idx: int = PAD_IDX,
) -> dict[str, Tensor]:
    """
    Pads a variable-length batch to the longest sequence in that batch
    (dynamic padding — shorter batches = fewer wasted PAD tokens).

    Constructs masks here on CPU so the GPU never touches padding logic.

    Args:
        batch:   List of (src_ids, tgt_ids) from `TranslationDataset`.
        pad_idx: Index of the [PAD] token.

    Returns:
        Dictionary with keys:
            "src"      : (B, src_len)        — padded source ids
            "tgt_in"   : (B, tgt_len)        — target ids without last token
                         (teacher-forced decoder input)
            "tgt_out"  : (B, tgt_len)        — target ids without first token
                         (gold labels for cross-entropy loss)
            "src_mask" : (B, 1, 1, src_len)  — PAD mask for encoder
            "tgt_mask" : (B, 1, tgt_len, tgt_len) — causal + PAD mask for decoder
            "n_tokens" : scalar int           — number of non-PAD target tokens
                         (used for loss normalisation)
    """
    src_list, tgt_list = zip(*batch)

    # --- Pad source sequences ---
    max_src = max(len(s) for s in src_list)
    src_padded = [
        s + [pad_idx] * (max_src - len(s)) for s in src_list
    ]

    # --- Pad target sequences ---
    max_tgt = max(len(t) for t in tgt_list)
    tgt_padded = [
        t + [pad_idx] * (max_tgt - len(t)) for t in tgt_list
    ]

    src = torch.tensor(src_padded, dtype=torch.long)   # (B, src_len)
    tgt = torch.tensor(tgt_padded, dtype=torch.long)   # (B, tgt_len)

    # Teacher forcing: decoder input = tgt[:-1], labels = tgt[1:]
    tgt_in  = tgt[:, :-1]  # (B, tgt_len-1)
    tgt_out = tgt[:, 1:]   # (B, tgt_len-1)

    # --- Masks (see model.py for convention: True = masked out) ---
    src_mask = _make_src_mask(src, pad_idx)
    tgt_mask = _make_tgt_mask(tgt_in, pad_idx)

    # Count non-PAD tokens for loss normalisation
    n_tokens = (tgt_out != pad_idx).sum().item()

    return {
        "src":      src,
        "tgt_in":   tgt_in,
        "tgt_out":  tgt_out,
        "src_mask": src_mask,
        "tgt_mask": tgt_mask,
        "n_tokens": n_tokens,
    }


def _make_src_mask(src: Tensor, pad_idx: int) -> Tensor:
    """(B, src_len) → (B, 1, 1, src_len)  True where PAD."""
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def _make_tgt_mask(tgt: Tensor, pad_idx: int) -> Tensor:
    """
    (B, tgt_len) → (B, 1, tgt_len, tgt_len)
    Combines causal (look-ahead) mask with PAD mask.
    """
    tgt_len = tgt.size(1)
    causal = torch.triu(
        torch.ones(tgt_len, tgt_len, dtype=torch.bool),
        diagonal=1,
    )  # (tgt_len, tgt_len)
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, tgt_len)
    return causal.unsqueeze(0).unsqueeze(0) | pad_mask      # (B, 1, tgt_len, tgt_len)


# ---------------------------------------------------------------------------
# Step 5 — DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    dataset_dict: Any,
    tokenizer: TokenizerWrapper,
    batch_size: int = 128,
    max_src_len: int = 128,
    max_tgt_len: int = 128,
    num_workers: int = 4,
    extra_train_pairs: list[dict[str, str]] | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Constructs train, validation, and test DataLoaders.

    Val and test always come from Multi30k for consistent evaluation.
    Extra training pairs (from additional datasets) are merged into train.

    Args:
        dataset_dict:      HuggingFace DatasetDict with train/validation/test splits.
        tokenizer:         Fitted TokenizerWrapper.
        batch_size:        Number of sentence pairs per batch.
        max_src_len:       Source truncation length.
        max_tgt_len:       Target truncation length.
        num_workers:       DataLoader worker processes (0 = main process only).
        extra_train_pairs: Additional {"en", "de"} dicts merged into train split.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    _collate = lambda batch: collate_fn(batch, pad_idx=tokenizer.pad_idx)
    pin     = torch.cuda.is_available()
    persist = num_workers > 0

    print("Building datasets ...")

    # Combine Multi30k train with any extra pairs
    train_examples = list(dataset_dict["train"])
    if extra_train_pairs:
        train_examples = train_examples + extra_train_pairs
        print(f"  Combined train: {len(dataset_dict['train']):,} Multi30k "
              f"+ {len(extra_train_pairs):,} extra = {len(train_examples):,} total")

    train_ds = TranslationDataset(train_examples,             tokenizer, max_src_len, max_tgt_len)
    val_ds   = TranslationDataset(dataset_dict["validation"], tokenizer, max_src_len, max_tgt_len)
    test_ds  = TranslationDataset(dataset_dict["test"],       tokenizer, max_src_len, max_tgt_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persist,
        drop_last=True,   # keeps batch size fixed; avoids tiny final batch issues
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persist,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persist,
    )

    print(
        f"DataLoaders ready — "
        f"train={len(train_loader)} batches, "
        f"val={len(val_loader)} batches, "
        f"test={len(test_loader)} batches  "
        f"(pin_memory={pin}, workers={num_workers})"
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# End-to-end setup helper (called once before training)
# ---------------------------------------------------------------------------

def prepare_data(
    data_dir: str | Path = "data",
    tokenizer_dir: str | Path = "data/tokenizer",
    vocab_size: int = 16_000,
    batch_size: int = 128,
    max_src_len: int = 128,
    max_tgt_len: int = 128,
    num_workers: int = 4,
    force_retrain_tokenizer: bool = False,
    extra_dataset: str | None = None,
    extra_max: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, TokenizerWrapper]:
    """
    One-shot setup: download data → (optionally train tokenizer) → build loaders.

    Args:
        data_dir:                Root data directory.
        tokenizer_dir:           Where to save/load the BPE tokenizer.
        vocab_size:              BPE vocab size (used only when training).
        batch_size:              Samples per batch.
        max_src_len:             Source truncation length in tokens.
        max_tgt_len:             Target truncation length in tokens.
        num_workers:             DataLoader worker processes.
        force_retrain_tokenizer: Re-run BPE training even if tokenizer exists.
        extra_dataset:           Short name of an additional dataset to merge
                                 into training ("opus_books", "wmt14", "tatoeba").
        extra_max:               Max pairs to load from extra dataset (None = all).

    Returns:
        (train_loader, val_loader, test_loader, tokenizer)
    """
    data_dir      = Path(data_dir)
    tokenizer_dir = Path(tokenizer_dir)
    tokenizer_file = tokenizer_dir / "tokenizer.json"

    # 1. Download Multi30k
    dataset = download_multi30k(cache_dir=data_dir / "raw")

    # 2. Optionally download extra dataset
    extra_pairs: list[dict[str, str]] | None = None
    if extra_dataset:
        extra_pairs = download_extra_dataset(
            extra_dataset, max_examples=extra_max, cache_dir=data_dir / "raw"
        )

    # 3. Train tokenizer (skip if already exists)
    if force_retrain_tokenizer or not tokenizer_file.exists():
        corpus_path = build_corpus_file(
            dataset,
            output_path=data_dir / "raw" / "combined_corpus.txt",
            extra_pairs=extra_pairs,
        )
        def _iter_corpus():
            with corpus_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line

        tokenizer = train_tokenizer(
            corpus_iterator=_iter_corpus(),
            save_dir=tokenizer_dir,
            vocab_size=vocab_size,
        )
    else:
        print(f"Loading existing tokenizer from {tokenizer_dir} ...")
        tokenizer = load_tokenizer(tokenizer_dir)

    print(tokenizer)

    # 4. Build DataLoaders
    train_loader, val_loader, test_loader = build_dataloaders(
        dataset_dict=dataset,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
        num_workers=num_workers,
        extra_train_pairs=extra_pairs,
    )

    return train_loader, val_loader, test_loader, tokenizer


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_loader, val_loader, test_loader, tokenizer = prepare_data(
        batch_size=4,
        num_workers=0,   # 0 for Windows compatibility in __main__
        force_retrain_tokenizer=False,
    )

    batch = next(iter(train_loader))
    print("\nSample batch:")
    for key, val in batch.items():
        if isinstance(val, Tensor):
            print(f"  {key:10s} {tuple(val.shape)}  dtype={val.dtype}")
        else:
            print(f"  {key:10s} {val}")

    # Decode first example
    src_ids  = batch["src"][0].tolist()
    tgt_ids  = batch["tgt_out"][0].tolist()
    print("\nDecoded src :", tokenizer.decode(src_ids))
    print("Decoded tgt :", tokenizer.decode(tgt_ids))
