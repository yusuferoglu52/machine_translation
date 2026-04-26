"""
Training loop for the EN→DE Transformer NMT system.

Features:
    - Automatic Mixed Precision (torch.amp) — FP16 forward/backward on RTX 3060
      Tensor Cores, FP32 master weights and optimizer state.
    - Noam learning rate schedule  (Vaswani et al., 2017 §5.3)
    - Label-smoothed cross-entropy loss
    - Gradient clipping (prevents exploding gradients in early training)
    - Per-epoch validation loss + corpus BLEU tracking
    - Checkpoint saving: best-BLEU model and periodic epoch snapshots
    - Graceful Ctrl-C interruption (saves checkpoint before exiting)

Usage:
    python src/train.py                         # default hyperparameters
    python src/train.py --epochs 30 --batch-size 64
    python src/train.py --resume checkpoints/best.pt
"""

from __future__ import annotations

# datasets must be imported before torch to avoid a segfault on Windows
# caused by a conflict between its C extensions and the CUDA runtime.
import datasets  # noqa: F401  — must load before torch to avoid CUDA segfault on Windows

import argparse
import math
import os
import signal
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim import Adam

# sacrebleu is the standard BLEU implementation used in MT research
try:
    from sacrebleu.metrics import BLEU as SacreBLEU
    _SACREBLEU = True
except ImportError:
    _SACREBLEU = False
    print("[warn] sacrebleu not installed — BLEU evaluation disabled. pip install sacrebleu")

# Local modules
sys.path.insert(0, str(Path(__file__).parent))
from model import Seq2SeqTransformer, count_parameters
from dataset import prepare_data
from tokenizer import TokenizerWrapper


# ---------------------------------------------------------------------------
# Noam Learning Rate Scheduler
# ---------------------------------------------------------------------------

class NoamScheduler:
    """
    Implements the Noam learning rate schedule from Vaswani et al., 2017:

        lr = d_model^{-0.5} · min(step^{-0.5}, step · warmup_steps^{-1.5})

    The schedule linearly warms up for `warmup_steps` steps then decays
    proportional to the inverse square root of the step number.

    This is applied directly to the optimizer's param groups rather than
    using torch.optim.lr_scheduler so we have full control over step counting.

    Args:
        optimizer:     The Adam optimizer instance.
        d_model:       Model dimensionality (used as the scale factor).
        warmup_steps:  Number of linear warm-up steps (paper uses 4000).
        factor:        Additional multiplicative scale (default 1.0).
    """

    def __init__(
        self,
        optimizer: Adam,
        d_model: int,
        warmup_steps: int = 4000,
        factor: float = 1.0,
    ) -> None:
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self._step = 0

    def step(self) -> float:
        """Advance one step and update optimizer learning rate. Returns new lr."""
        self._step += 1
        lr = self._compute_lr(self._step)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def _compute_lr(self, step: int) -> float:
        if step == 0:
            return 0.0
        return self.factor * (
            self.d_model ** -0.5
            * min(step ** -0.5, step * self.warmup_steps ** -1.5)
        )

    @property
    def current_lr(self) -> float:
        return self._compute_lr(self._step)

    def state_dict(self) -> dict:
        return {"step": self._step}

    def load_state_dict(self, state: dict) -> None:
        self._step = state["step"]
        # Restore lr so the optimizer is in the correct state after resuming
        for pg in self.optimizer.param_groups:
            pg["lr"] = self.current_lr


# ---------------------------------------------------------------------------
# Label-Smoothed Cross-Entropy Loss
# ---------------------------------------------------------------------------

class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy loss with label smoothing (Szegedy et al., 2016).

    Instead of a hard one-hot target, distributes ε/(V-1) probability mass
    to all non-target tokens, keeping (1 - ε) on the correct token:

        y_smooth[i] = ε / (V - 1)   if i ≠ target
        y_smooth[i] = 1 - ε          if i == target

    This prevents the model from becoming over-confident and improves
    generalisation — standard in all competitive MT systems.

    PAD positions are excluded from the loss (zero weight).

    Args:
        vocab_size:  Size of the output vocabulary.
        pad_idx:     Index of the [PAD] token (excluded from loss).
        smoothing:   Label smoothing coefficient ε (paper uses 0.1).
    """

    def __init__(
        self, vocab_size: int, pad_idx: int, smoothing: float = 0.1
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B * tgt_len, vocab_size) — raw model output (pre-softmax)
            targets: (B * tgt_len,)            — gold token ids

        Returns:
            Scalar mean loss over non-PAD positions.
        """
        # Cast to FP32: custom loss functions must opt out of AMP manually
        # because autocast would keep logits in FP16, causing precision loss
        # in the vocab-size summation (16000 subnormal values * log_prob).
        logits = logits.float()

        # Build smoothed target distribution (always FP32)
        with torch.no_grad():
            smooth_targets = torch.full_like(
                logits, self.smoothing / (self.vocab_size - 2)
            )  # -2 for correct class and PAD
            smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)
            smooth_targets[:, self.pad_idx] = 0.0  # PAD gets zero weight

        # Mask out PAD positions entirely
        pad_mask = (targets == self.pad_idx)
        smooth_targets[pad_mask] = 0.0

        # KL divergence between smooth distribution and log-softmax output
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = -(smooth_targets * log_probs).sum(dim=-1)  # per-token loss

        # Mean over non-PAD tokens only
        non_pad = (~pad_mask).sum()
        return loss.sum() / non_pad.clamp(min=1)


# ---------------------------------------------------------------------------
# Greedy decode for BLEU evaluation (fast, no beam search)
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_decode_batch(
    model: Seq2SeqTransformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    tokenizer: TokenizerWrapper,
    max_len: int = 128,
    device: torch.device = torch.device("cpu"),
) -> list[list[int]]:
    """
    Greedy (argmax) decoding for a batch of source sentences.
    Used during validation for fast BLEU estimation.

    Returns a list of token-id lists (without [SOS]/[EOS]).
    """
    model.eval()
    batch_size = src.size(0)

    memory = model.encode(src, src_mask)  # (B, src_len, d_model)

    # Decoder input starts with [SOS] for every sentence in the batch
    ys = torch.full(
        (batch_size, 1), tokenizer.sos_idx, dtype=torch.long, device=device
    )
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        from model import make_tgt_mask
        tgt_mask = make_tgt_mask(ys, tokenizer.pad_idx).to(device)

        out = model.decode(ys, memory, tgt_mask, src_mask)  # (B, t, d_model)
        logits = model.output_projection(out[:, -1, :])     # (B, vocab)
        next_token = logits.argmax(dim=-1, keepdim=True)    # (B, 1)

        ys = torch.cat([ys, next_token], dim=1)

        finished |= (next_token.squeeze(1) == tokenizer.eos_idx)
        if finished.all():
            break

    # Strip [SOS] (position 0) and everything from [EOS] onward
    results = []
    eos = tokenizer.eos_idx
    for seq in ys[:, 1:].tolist():
        clean = seq[:seq.index(eos)] if eos in seq else seq
        results.append(clean)
    return results


# ---------------------------------------------------------------------------
# BLEU computation
# ---------------------------------------------------------------------------

def compute_bleu(
    model: Seq2SeqTransformer,
    loader,
    tokenizer: TokenizerWrapper,
    device: torch.device,
    max_batches: int | None = 20,
) -> float:
    """
    Computes corpus BLEU on up to `max_batches` batches from `loader`.

    Using sacrebleu with `tokenize="13a"` (standard MT tokenization) for
    results comparable to published Multi30k benchmarks.

    Args:
        max_batches: Cap batches to keep validation fast (~20 = ~2560 sentences).
                     Pass None to evaluate the entire split.

    Returns:
        BLEU score (0–100).
    """
    if not _SACREBLEU:
        return 0.0

    hypotheses: list[str] = []
    references: list[str] = []

    model.eval()
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        src      = batch["src"].to(device, non_blocking=True)
        src_mask = batch["src_mask"].to(device, non_blocking=True)
        tgt_out  = batch["tgt_out"]  # stays on CPU for reference decoding

        hyp_ids = greedy_decode_batch(model, src, src_mask, tokenizer, device=device)
        ref_ids = tgt_out.tolist()

        for hyp, ref in zip(hyp_ids, ref_ids):
            hypotheses.append(tokenizer.decode(hyp, skip_special_tokens=True))
            references.append(tokenizer.decode(ref, skip_special_tokens=True))

    bleu = SacreBLEU(tokenize="13a")
    score = bleu.corpus_score(hypotheses, [references])
    return score.score


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def train_epoch(
    model: Seq2SeqTransformer,
    loader,
    criterion: LabelSmoothingLoss,
    optimizer: Adam,
    scheduler: NoamScheduler,
    scaler: GradScaler,
    device: torch.device,
    clip_grad: float = 1.0,
    log_interval: int = 50,
    accum_steps: int = 1,
) -> tuple[float, float]:
    """
    Runs one full pass over the training DataLoader with AMP.

    Args:
        clip_grad:    Max gradient norm (applied before optimizer step).
        log_interval: Print a progress line every N batches.
        accum_steps:  Gradient accumulation steps. Effective batch size =
                      batch_size * accum_steps without extra VRAM cost.
    Returns:
        (mean_loss, final_lr) for the epoch.
    """
    model.train()
    total_loss   = 0.0
    total_tokens = 0
    lr           = scheduler.current_lr
    start        = time.perf_counter()

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(loader):
        src      = batch["src"].to(device, non_blocking=True)
        tgt_in   = batch["tgt_in"].to(device, non_blocking=True)
        tgt_out  = batch["tgt_out"].to(device, non_blocking=True)
        src_mask = batch["src_mask"].to(device, non_blocking=True)
        tgt_mask = batch["tgt_mask"].to(device, non_blocking=True)
        n_tokens = batch["n_tokens"]

        # AMP forward pass
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            logits = model(src, tgt_in, src_mask, tgt_mask)
            loss   = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_out.reshape(-1),
            )
            # Scale loss so gradients accumulate correctly across micro-batches
            loss = loss / accum_steps

        scaler.scale(loss).backward()

        is_update_step = ((batch_idx + 1) % accum_steps == 0) or \
                         (batch_idx + 1 == len(loader))

        if is_update_step:
            lr = scheduler.step()  # Noam lr set BEFORE optimizer uses it
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss   += loss.item() * accum_steps * n_tokens
        total_tokens += n_tokens

        if (batch_idx + 1) % log_interval == 0:
            elapsed     = time.perf_counter() - start
            avg_loss    = total_loss / max(total_tokens, 1)
            tok_per_sec = total_tokens / elapsed
            print(
                f"  [{batch_idx + 1:>4}/{len(loader)}]  "
                f"loss={avg_loss:.4f}  "
                f"ppl={math.exp(min(avg_loss, 20)):.2f}  "
                f"lr={lr:.2e}  "
                f"tok/s={tok_per_sec:,.0f}"
            )

    mean_loss = total_loss / max(total_tokens, 1)
    return mean_loss, scheduler.current_lr


# ---------------------------------------------------------------------------
# One validation epoch
# ---------------------------------------------------------------------------

@torch.no_grad()
def val_epoch(
    model: Seq2SeqTransformer,
    loader,
    criterion: LabelSmoothingLoss,
    device: torch.device,
) -> float:
    """Runs one pass over the validation DataLoader. Returns mean loss."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in loader:
        src      = batch["src"].to(device, non_blocking=True)
        tgt_in   = batch["tgt_in"].to(device, non_blocking=True)
        tgt_out  = batch["tgt_out"].to(device, non_blocking=True)
        src_mask = batch["src_mask"].to(device, non_blocking=True)
        tgt_mask = batch["tgt_mask"].to(device, non_blocking=True)
        n_tokens = batch["n_tokens"]

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            logits = model(src, tgt_in, src_mask, tgt_mask)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_out.reshape(-1),
            )

        total_loss   += loss.item() * n_tokens
        total_tokens += n_tokens

    return total_loss / max(total_tokens, 1)


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    model: Seq2SeqTransformer,
    optimizer: Adam,
    scheduler: NoamScheduler,
    scaler: GradScaler,
    epoch: int,
    best_bleu: float,
    cfg: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch":        epoch,
            "best_bleu":    best_bleu,
            "model_state":  model.state_dict(),
            "optim_state":  optimizer.state_dict(),
            "sched_state":  scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "cfg":          cfg,
        },
        path,
    )
    print(f"  Checkpoint saved -> {path}")


def load_checkpoint(
    path: Path,
    model: Seq2SeqTransformer,
    optimizer: Adam,
    scheduler: NoamScheduler,
    scaler: GradScaler,
    device: torch.device,
) -> tuple[int, float]:
    """Returns (start_epoch, best_bleu)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optim_state"])
    scheduler.load_state_dict(ckpt["sched_state"])
    scaler.load_state_dict(ckpt["scaler_state"])
    print(f"Resumed from {path}  (epoch {ckpt['epoch']}, best BLEU {ckpt['best_bleu']:.2f})")
    return ckpt["epoch"] + 1, ckpt["best_bleu"]


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train(cfg: dict) -> None:
    """
    Full training loop.

    Args:
        cfg: Hyperparameter dictionary (populated from CLI args below).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"VRAM   : {props.total_memory / 1e9:.1f} GB")
        print(f"Compute: sm_{props.major}{props.minor}")

    # ── Data ───────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, tokenizer = prepare_data(
        data_dir=cfg["data_dir"],
        tokenizer_dir=cfg["tokenizer_dir"],
        vocab_size=cfg["vocab_size"],
        batch_size=cfg["batch_size"],
        max_src_len=cfg["max_src_len"],
        max_tgt_len=cfg["max_tgt_len"],
        num_workers=cfg["num_workers"],
        force_retrain_tokenizer=cfg["retrain_tokenizer"],
        extra_dataset=cfg["extra_dataset"] or None,
        extra_max=cfg["extra_max"],
    )

    # ── Model ──────────────────────────────────────────────────────────────
    model = Seq2SeqTransformer(
        src_vocab_size=tokenizer.vocab_size,
        tgt_vocab_size=tokenizer.vocab_size,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_encoder_layers=cfg["n_layers"],
        n_decoder_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
        share_embeddings=True,
    ).to(device)

    print(f"\nParameters : {count_parameters(model):,}")

    # ── Optimizer + Scheduler ──────────────────────────────────────────────
    # Adam betas from the paper: β1=0.9, β2=0.98, ε=1e-9
    optimizer = Adam(
        model.parameters(),
        lr=1.0,             # actual lr is fully controlled by NoamScheduler
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler = NoamScheduler(
        optimizer,
        d_model=cfg["d_model"],
        warmup_steps=cfg["warmup_steps"],
    )

    # ── Loss ───────────────────────────────────────────────────────────────
    criterion = LabelSmoothingLoss(
        vocab_size=tokenizer.vocab_size,
        pad_idx=tokenizer.pad_idx,
        smoothing=cfg["label_smoothing"],
    )

    # ── AMP GradScaler ─────────────────────────────────────────────────────
    # enabled=False on CPU (scaler is a no-op but keeps code uniform)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # ── Optional Resume ────────────────────────────────────────────────────
    start_epoch = 1
    best_bleu   = 0.0
    ckpt_dir    = Path(cfg["checkpoint_dir"])

    if cfg["resume"]:
        resume_path = Path(cfg["resume"])
        start_epoch, best_bleu = load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler, device
        )

    # ── Graceful Ctrl-C ────────────────────────────────────────────────────
    _interrupted = False
    def _handle_sigint(sig, frame):
        nonlocal _interrupted
        print("\n[!] Interrupted — saving checkpoint before exit ...")
        _interrupted = True
    signal.signal(signal.SIGINT, _handle_sigint)

    # ── Training Loop ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f" Training for {cfg['epochs']} epochs  |  batch={cfg['batch_size']}  "
          f"accum={cfg['accum_steps']}  (eff. batch={cfg['batch_size']*cfg['accum_steps']})")
    print(f" d_model={cfg['d_model']}  n_heads={cfg['n_heads']}  "
          f"n_layers={cfg['n_layers']}  d_ff={cfg['d_ff']}")
    print(f" warmup={cfg['warmup_steps']}  label_smooth={cfg['label_smoothing']}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        epoch_start = time.perf_counter()
        print(f"Epoch {epoch}/{cfg['epochs']}")

        # Train
        train_loss, lr = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler,
            device, clip_grad=cfg["clip_grad"], log_interval=cfg["log_interval"],
            accum_steps=cfg["accum_steps"],
        )

        # Validate
        val_loss = val_epoch(model, val_loader, criterion, device)

        # BLEU (greedy, capped at 20 batches for speed)
        bleu = compute_bleu(
            model, val_loader, tokenizer, device,
            max_batches=cfg["bleu_batches"],
        )

        elapsed = time.perf_counter() - epoch_start
        print(
            f"  -> train_loss={train_loss:.4f}  "
            f"ppl={math.exp(min(train_loss, 20)):.2f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_ppl={math.exp(min(val_loss, 20)):.2f}  "
            f"BLEU={bleu:.2f}  "
            f"lr={lr:.2e}  "
            f"time={elapsed:.1f}s"
        )

        # Save best checkpoint
        if bleu > best_bleu:
            best_bleu = bleu
            save_checkpoint(
                ckpt_dir / "best.pt",
                model, optimizer, scheduler, scaler, epoch, best_bleu, cfg,
            )
            print(f"  ★ New best BLEU: {best_bleu:.2f}")

        # Periodic epoch checkpoint
        if epoch % cfg["save_every"] == 0:
            save_checkpoint(
                ckpt_dir / f"epoch_{epoch:03d}.pt",
                model, optimizer, scheduler, scaler, epoch, best_bleu, cfg,
            )

        if _interrupted:
            save_checkpoint(
                ckpt_dir / "interrupted.pt",
                model, optimizer, scheduler, scaler, epoch, best_bleu, cfg,
            )
            print("Checkpoint saved. Exiting.")
            sys.exit(0)

    # ── Final test-set BLEU (always from best checkpoint, not last epoch) ────
    print(f"\n{'='*60}")
    print("Evaluating on test set ...")
    best_ckpt = ckpt_dir / "best.pt"
    if best_ckpt.exists():
        load_checkpoint(best_ckpt, model, optimizer, scheduler, scaler, device)
        print(f"Loaded best checkpoint for test evaluation.")
    test_bleu = compute_bleu(model, test_loader, tokenizer, device, max_batches=None)
    print(f"Test BLEU (greedy): {test_bleu:.2f}")
    print(f"Best val BLEU achieved: {best_bleu:.2f}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> dict:
    p = argparse.ArgumentParser(description="Train the EN→DE Transformer NMT model.")

    # Data
    p.add_argument("--data-dir",        default="data",           type=str)
    p.add_argument("--tokenizer-dir",   default="data/tokenizer", type=str)
    p.add_argument("--checkpoint-dir",  default="checkpoints",    type=str)
    p.add_argument("--resume",          default="",               type=str,
                   help="Path to checkpoint to resume from.")
    p.add_argument("--retrain-tokenizer", action="store_true",
                   help="Force re-training the BPE tokenizer.")
    p.add_argument("--extra-dataset", default="", type=str,
                   help="Extra HF dataset to merge into training: opus_books | wmt14 | tatoeba")
    p.add_argument("--extra-max", default=None, type=int,
                   help="Max pairs to load from extra dataset (default: all).")

    # Model
    p.add_argument("--d-model",  default=512,  type=int)
    p.add_argument("--n-heads",  default=8,    type=int)
    p.add_argument("--n-layers", default=6,    type=int)
    p.add_argument("--d-ff",     default=2048, type=int)
    p.add_argument("--dropout",  default=0.1,  type=float)

    # Tokenizer / Data
    p.add_argument("--vocab-size",   default=16_000, type=int)
    p.add_argument("--max-src-len",  default=128,    type=int)
    p.add_argument("--max-tgt-len",  default=128,    type=int)
    p.add_argument("--num-workers",  default=4,      type=int)

    # Training
    p.add_argument("--epochs",          default=30,   type=int)
    p.add_argument("--batch-size",      default=128,  type=int)
    p.add_argument("--accum-steps",     default=4,    type=int,
                   help="Gradient accumulation steps. Effective batch = batch_size * accum_steps.")
    p.add_argument("--warmup-steps",    default=4000, type=int)
    p.add_argument("--label-smoothing", default=0.1,  type=float)
    p.add_argument("--clip-grad",       default=1.0,  type=float)

    # Logging
    p.add_argument("--log-interval",  default=50,  type=int,
                   help="Print progress every N batches.")
    p.add_argument("--save-every",    default=5,   type=int,
                   help="Save epoch checkpoint every N epochs.")
    p.add_argument("--bleu-batches",  default=20,  type=int,
                   help="Number of val batches to use for BLEU (None=all).")

    args = p.parse_args()
    return {
        "data_dir":          args.data_dir,
        "tokenizer_dir":     args.tokenizer_dir,
        "checkpoint_dir":    args.checkpoint_dir,
        "resume":            args.resume,
        "retrain_tokenizer": args.retrain_tokenizer,
        "extra_dataset":     args.extra_dataset,
        "extra_max":         args.extra_max,
        "d_model":           args.d_model,
        "n_heads":           args.n_heads,
        "n_layers":          args.n_layers,
        "d_ff":              args.d_ff,
        "dropout":           args.dropout,
        "vocab_size":        args.vocab_size,
        "max_src_len":       args.max_src_len,
        "max_tgt_len":       args.max_tgt_len,
        "num_workers":       args.num_workers,
        "epochs":            args.epochs,
        "batch_size":        args.batch_size,
        "accum_steps":       args.accum_steps,
        "warmup_steps":      args.warmup_steps,
        "label_smoothing":   args.label_smoothing,
        "clip_grad":         args.clip_grad,
        "log_interval":      args.log_interval,
        "save_every":        args.save_every,
        "bleu_batches":      args.bleu_batches,
    }


if __name__ == "__main__":
    cfg = _parse_args()
    train(cfg)
