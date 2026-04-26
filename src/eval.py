"""Quick BLEU evaluation of best.pt on the test set using the fixed decoder."""
import datasets  # noqa: F401 — must be before torch on Windows
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from dataset import prepare_data
from model import Seq2SeqTransformer, make_src_mask
from tokenizer import TokenizerWrapper

try:
    from sacrebleu.metrics import BLEU as SacreBLEU
except ImportError:
    print("pip install sacrebleu")
    sys.exit(1)

CKPT = Path("checkpoints/best.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
cfg  = ckpt["cfg"]

_, _, test_loader, tokenizer = prepare_data(batch_size=64, num_workers=0)

model = Seq2SeqTransformer(
    src_vocab_size=tokenizer.vocab_size,
    tgt_vocab_size=tokenizer.vocab_size,
    d_model=cfg["d_model"],
    n_heads=cfg["n_heads"],
    n_encoder_layers=cfg["n_layers"],
    n_decoder_layers=cfg["n_layers"],
    d_ff=cfg["d_ff"],
    dropout=cfg["dropout"],
)
model.load_state_dict(ckpt["model_state"])
model.to(DEVICE).eval()
print(f"Loaded epoch={ckpt['epoch']}  best_val_BLEU={ckpt['best_bleu']:.2f}")

hypotheses, references = [], []

with torch.no_grad():
    for batch in test_loader:
        src      = batch["src"].to(DEVICE)
        src_mask = batch["src_mask"].to(DEVICE)
        tgt_out  = batch["tgt_out"]

        memory = model.encode(src, src_mask)
        batch_size = src.size(0)
        ys = torch.full((batch_size, 1), tokenizer.sos_idx, dtype=torch.long, device=DEVICE)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=DEVICE)

        for _ in range(128):
            from model import make_tgt_mask
            tgt_mask = make_tgt_mask(ys, tokenizer.pad_idx).to(DEVICE)
            out    = model.decode(ys, memory, tgt_mask, src_mask)
            logits = model.output_projection(out[:, -1, :])
            next_t = logits.argmax(dim=-1, keepdim=True)
            ys     = torch.cat([ys, next_t], dim=1)
            finished |= (next_t.squeeze(1) == tokenizer.eos_idx)
            if finished.all():
                break

        eos = tokenizer.eos_idx
        for hyp_seq, ref_seq in zip(ys[:, 1:].tolist(), tgt_out.tolist()):
            clean_hyp = hyp_seq[:hyp_seq.index(eos)] if eos in hyp_seq else hyp_seq
            hypotheses.append(tokenizer.decode(clean_hyp))
            references.append(tokenizer.decode(ref_seq))

bleu = SacreBLEU(tokenize="13a")
score = bleu.corpus_score(hypotheses, [references])
print(f"\nTest BLEU (greedy, fixed decoder): {score.score:.2f}")
print(f"\nSample outputs:")
for en_ids, hyp, ref in zip(list(test_loader)[0]["src"][:3].tolist(),
                             hypotheses[:3], references[:3]):
    print(f"  HYP: {hyp}")
    print(f"  REF: {ref}")
    print()
