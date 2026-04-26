"""
Inference script for the EN→DE Transformer NMT system.

Implements Beam Search decoding (width=5) with length penalty — the standard
high-quality decoding strategy for NMT, significantly outperforming greedy
on longer sentences.

Modes:
    --interactive   Read sentences from stdin one at a time (REPL)
    --input FILE    Translate every line of a file, write to --output
    --sentence STR  Translate a single sentence and print the result

Usage:
    python src/translate.py --checkpoint checkpoints/best.pt --interactive
    python src/translate.py --checkpoint checkpoints/best.pt --sentence "Two dogs are playing."
    python src/translate.py --checkpoint checkpoints/best.pt --input data/test.en --output out.de
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from model import Seq2SeqTransformer, make_src_mask, make_tgt_mask
from tokenizer import load_tokenizer, TokenizerWrapper


# ---------------------------------------------------------------------------
# Beam Search — data structures
# ---------------------------------------------------------------------------

@dataclass(order=True)
class Hypothesis:
    """
    A single beam hypothesis.

    Sorting is by `score` (highest first when used with a max-heap via negation,
    or lowest first naturally — callers sort explicitly).

    Attributes:
        score:      Accumulated log-probability (un-length-penalised).
        token_ids:  Sequence of decoded token ids including [SOS].
        done:       True if [EOS] has been emitted.
    """
    score: float
    token_ids: list[int] = field(compare=False)
    done: bool = field(default=False, compare=False)

    def length_penalised_score(self, alpha: float) -> float:
        """
        Apply Wu et al. (2016) length penalty:

            lp(Y) = (5 + |Y|)^α / (5 + 1)^α

        Higher α (0–1) increasingly rewards longer hypotheses, preventing
        beam search from preferring short, low-perplexity translations.
        """
        lp = ((5.0 + len(self.token_ids)) / 6.0) ** alpha
        return self.score / lp


# ---------------------------------------------------------------------------
# Beam Search decoder
# ---------------------------------------------------------------------------

@torch.no_grad()
def beam_search(
    model: Seq2SeqTransformer,
    src_ids: list[int],
    tokenizer: TokenizerWrapper,
    device: torch.device,
    beam_width: int = 5,
    max_len: int = 128,
    length_penalty: float = 0.6,
    min_len: int = 1,
) -> list[tuple[str, float]]:
    """
    Beam search decoding for a single source sentence.

    Algorithm (per step t):
        1. For each active hypothesis h, run the decoder one step to get
           log P(y_t | y_{<t}, x).
        2. Expand each hypothesis with every vocabulary token →
           beam_width × vocab_size candidates.
        3. Keep the top beam_width candidates as the new beam.
        4. Move any hypothesis that emitted [EOS] to `completed`.
        5. Stop when all beams are completed or t == max_len.

    Efficiency note: all active hypotheses share the same encoder `memory`
    tensor, which is computed once and reused across all steps.

    Args:
        model:          Trained Seq2SeqTransformer (eval mode, on device).
        src_ids:        Token ids for the source sentence (including SOS/EOS).
        tokenizer:      TokenizerWrapper for special-token ids and decoding.
        device:         Torch device.
        beam_width:     Number of hypotheses to maintain (B).
        max_len:        Maximum output length (hard cap).
        length_penalty: α exponent for Wu et al. length penalty (0=none, 1=full).
        min_len:        Minimum output tokens before [EOS] is allowed.

    Returns:
        List of (decoded_string, lp_score) sorted by length-penalised score,
        best first. Length = beam_width (or fewer if early stopping).
    """
    model.eval()

    pad_idx = tokenizer.pad_idx
    sos_idx = tokenizer.sos_idx
    eos_idx = tokenizer.eos_idx
    vocab_size = tokenizer.vocab_size

    # ── Encode source once ────────────────────────────────────────────────
    src = torch.tensor([src_ids], dtype=torch.long, device=device)  # (1, src_len)
    src_mask = make_src_mask(src, pad_idx).to(device)               # (1,1,1,src_len)
    memory = model.encode(src, src_mask)                             # (1, src_len, d_model)

    # Expand memory for beam_width parallel hypotheses: (B, src_len, d_model)
    memory = memory.expand(beam_width, -1, -1)
    src_mask = src_mask.expand(beam_width, -1, -1, -1)

    # ── Initialise beam with the single [SOS] hypothesis ─────────────────
    active: list[Hypothesis] = [
        Hypothesis(score=0.0, token_ids=[sos_idx])
    ]
    completed: list[Hypothesis] = []

    for step in range(max_len):
        # Nothing left to expand
        if not active:
            break

        n_active = len(active)

        # Build decoder input tensor: (n_active, current_seq_len)
        tgt = torch.tensor(
            [h.token_ids for h in active], dtype=torch.long, device=device
        )
        tgt_mask = make_tgt_mask(tgt, pad_idx).to(device)  # causal mask

        # Slice memory/src_mask to match the actual number of active beams
        mem_slice  = memory[:n_active]
        smsk_slice = src_mask[:n_active]

        # Decoder forward — only the last position matters
        dec_out = model.decode(tgt, mem_slice, tgt_mask, smsk_slice)
        logits  = model.output_projection(dec_out[:, -1, :])  # (n_active, V)
        log_probs = F.log_softmax(logits, dim=-1)             # (n_active, V)

        # ── Expand each active hypothesis ─────────────────────────────────
        candidates: list[Hypothesis] = []

        for i, hyp in enumerate(active):
            token_log_probs = log_probs[i]  # (V,)

            # Suppress [EOS] until min_len tokens have been generated
            if step < min_len:
                token_log_probs = token_log_probs.clone()
                token_log_probs[eos_idx] = float("-inf")

            # Take top beam_width tokens per hypothesis
            top_scores, top_ids = token_log_probs.topk(beam_width)

            for score, token_id in zip(top_scores.tolist(), top_ids.tolist()):
                new_score    = hyp.score + score
                new_ids      = hyp.token_ids + [token_id]
                is_done      = (token_id == eos_idx)
                new_hyp      = Hypothesis(
                    score=new_score, token_ids=new_ids, done=is_done
                )

                if is_done:
                    completed.append(new_hyp)
                else:
                    candidates.append(new_hyp)

        # ── Prune: keep top beam_width by raw score ───────────────────────
        candidates.sort(key=lambda h: h.score, reverse=True)
        active = candidates[:beam_width]

        # Early exit if we have enough completed hypotheses
        if len(completed) >= beam_width:
            break

    # Anything still active at max_len is treated as completed
    completed.extend(active)

    # ── Rank by length-penalised score ────────────────────────────────────
    completed.sort(
        key=lambda h: h.length_penalised_score(length_penalty), reverse=True
    )

    results = []
    for hyp in completed[:beam_width]:
        # Strip [SOS] and [EOS] from output
        ids = hyp.token_ids[1:]  # remove leading [SOS]
        if eos_idx in ids:
            ids = ids[: ids.index(eos_idx)]
        text = tokenizer.decode(ids, skip_special_tokens=True)
        results.append((text, hyp.length_penalised_score(length_penalty)))

    return results


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: str | Path,
    tokenizer: TokenizerWrapper,
    device: torch.device,
) -> Seq2SeqTransformer:
    """
    Restores a Seq2SeqTransformer from a training checkpoint.

    The checkpoint's `cfg` dict is used to reconstruct the architecture so
    hyperparameters never need to be supplied again at inference time.

    Args:
        checkpoint_path: Path to a `.pt` checkpoint saved by `train.py`.
        tokenizer:       Fitted TokenizerWrapper (for vocab_size).
        device:          Torch device.

    Returns:
        Model in eval mode on the specified device.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg  = ckpt["cfg"]

    model = Seq2SeqTransformer(
        src_vocab_size=tokenizer.vocab_size,
        tgt_vocab_size=tokenizer.vocab_size,
        d_model=cfg.get("d_model", 512),
        n_heads=cfg.get("n_heads", 8),
        n_encoder_layers=cfg.get("n_layers", 6),
        n_decoder_layers=cfg.get("n_layers", 6),
        d_ff=cfg.get("d_ff", 2048),
        dropout=0.0,           # no dropout at inference
        share_embeddings=True,
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    epoch     = ckpt.get("epoch", "?")
    best_bleu = ckpt.get("best_bleu", 0.0)
    print(f"Loaded checkpoint: epoch={epoch}  best_BLEU={best_bleu:.2f}")
    return model


# ---------------------------------------------------------------------------
# High-level translate function (single sentence)
# ---------------------------------------------------------------------------

def translate(
    sentence: str,
    model: Seq2SeqTransformer,
    tokenizer: TokenizerWrapper,
    device: torch.device,
    beam_width: int = 5,
    max_len: int = 128,
    length_penalty: float = 0.6,
    n_best: int = 1,
    show_alternatives: bool = False,
) -> str:
    """
    Translates a single English sentence to German.

    Args:
        sentence:          Raw English input string.
        model:             Loaded, eval-mode Seq2SeqTransformer.
        tokenizer:         Fitted TokenizerWrapper.
        device:            Torch device.
        beam_width:        Beam search width.
        max_len:           Max decoded length in tokens.
        length_penalty:    Length penalty α.
        n_best:            Number of top translations to return (joined by newline).
        show_alternatives: If True, prints all beam hypotheses with scores.

    Returns:
        Best translation string (or top-n joined by newlines).
    """
    src_ids = tokenizer.encode(sentence)
    results = beam_search(
        model, src_ids, tokenizer, device,
        beam_width=beam_width, max_len=max_len, length_penalty=length_penalty,
    )

    if show_alternatives:
        print("\n  Beam hypotheses:")
        for rank, (text, score) in enumerate(results, 1):
            print(f"    [{rank}] score={score:.4f}  {text}")

    return "\n".join(text for text, _ in results[:n_best])


# ---------------------------------------------------------------------------
# Batch translation (file mode)
# ---------------------------------------------------------------------------

def translate_file(
    input_path: Path,
    output_path: Path,
    model: Seq2SeqTransformer,
    tokenizer: TokenizerWrapper,
    device: torch.device,
    beam_width: int = 5,
    max_len: int = 128,
    length_penalty: float = 0.6,
) -> None:
    """
    Translates every line of `input_path` and writes results to `output_path`.

    Prints a progress line every 100 sentences.

    Args:
        input_path:  Plain-text file, one English sentence per line.
        output_path: Destination file for German translations.
    """
    sentences = input_path.read_text(encoding="utf-8").splitlines()
    total     = len(sentences)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    with output_path.open("w", encoding="utf-8") as fout:
        for i, sent in enumerate(sentences, 1):
            sent = sent.strip()
            if not sent:
                fout.write("\n")
                continue

            translation = translate(
                sent, model, tokenizer, device,
                beam_width=beam_width, max_len=max_len, length_penalty=length_penalty,
            )
            fout.write(translation + "\n")

            if i % 100 == 0 or i == total:
                elapsed  = time.perf_counter() - start
                sent_per_sec = i / elapsed
                print(f"  {i}/{total} sentences  ({sent_per_sec:.1f} sent/s)")

    elapsed = time.perf_counter() - start
    print(f"\nDone. {total} sentences in {elapsed:.1f}s -> {output_path}")


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

def interactive_loop(
    model: Seq2SeqTransformer,
    tokenizer: TokenizerWrapper,
    device: torch.device,
    beam_width: int = 5,
    max_len: int = 128,
    length_penalty: float = 0.6,
    show_alternatives: bool = False,
) -> None:
    """
    Reads English sentences from stdin and prints German translations.

    Type 'q' or press Ctrl-C / Ctrl-D to exit.
    """
    print("\nEN->DE Interactive Translator  (beam_width={}, alpha={})".format(
        beam_width, length_penalty
    ))
    print("Type an English sentence and press Enter. 'q' to quit.\n")

    while True:
        try:
            raw = input("EN > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if raw.lower() in {"q", "quit", "exit"}:
            print("Goodbye.")
            break
        if not raw:
            continue

        t0 = time.perf_counter()
        translation = translate(
            raw, model, tokenizer, device,
            beam_width=beam_width, max_len=max_len, length_penalty=length_penalty,
            show_alternatives=show_alternatives,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        print(f"DE > {translation}  ({elapsed_ms:.0f}ms)\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EN→DE NMT inference with beam search.")

    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/best.pt"),
        help="Path to trained model checkpoint (default: checkpoints/best.pt).",
    )
    p.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path("data/tokenizer"),
        help="Directory containing tokenizer.json.",
    )
    p.add_argument(
        "--beam-width",
        type=int,
        default=5,
        help="Beam search width (default: 5).",
    )
    p.add_argument(
        "--max-len",
        type=int,
        default=128,
        help="Maximum output length in tokens (default: 128).",
    )
    p.add_argument(
        "--length-penalty",
        type=float,
        default=0.6,
        help="Length penalty α, 0=none, 1=full (default: 0.6).",
    )

    # Mutually exclusive modes
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive REPL (default if no mode is given).",
    )
    mode.add_argument(
        "--sentence",
        type=str,
        default="",
        help="Translate a single sentence and exit.",
    )
    mode.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input file path (one sentence per line).",
    )

    p.add_argument(
        "--output",
        type=Path,
        default=Path("translations.de"),
        help="Output file path for --input mode (default: translations.de).",
    )
    p.add_argument(
        "--n-best",
        type=int,
        default=1,
        help="Number of top hypotheses to output in --sentence mode (default: 1).",
    )
    p.add_argument(
        "--show-alternatives",
        action="store_true",
        help="Print all beam hypotheses with scores (--sentence and --interactive).",
    )

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load tokenizer ─────────────────────────────────────────────────────
    tokenizer = load_tokenizer(args.tokenizer_dir)
    print(tokenizer)

    # ── Load model ─────────────────────────────────────────────────────────
    if not args.checkpoint.exists():
        sys.exit(
            f"[error] Checkpoint not found: {args.checkpoint}\n"
            "Train the model first with: python src/train.py"
        )
    model = load_model(args.checkpoint, tokenizer, device)

    # ── Dispatch to mode ───────────────────────────────────────────────────
    kwargs = dict(
        beam_width=args.beam_width,
        max_len=args.max_len,
        length_penalty=args.length_penalty,
    )

    if args.sentence:
        result = translate(
            args.sentence, model, tokenizer, device,
            n_best=args.n_best,
            show_alternatives=args.show_alternatives,
            **kwargs,
        )
        print(f"\nEN: {args.sentence}")
        print(f"DE: {result}")

    elif args.input is not None:
        if not args.input.exists():
            sys.exit(f"[error] Input file not found: {args.input}")
        translate_file(args.input, args.output, model, tokenizer, device, **kwargs)

    else:
        # Default: interactive
        interactive_loop(
            model, tokenizer, device,
            show_alternatives=args.show_alternatives,
            **kwargs,
        )


if __name__ == "__main__":
    main()
