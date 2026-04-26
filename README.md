# Transformer NMT — English → German

<p align="center">
  <img src="https://img.shields.io/badge/BLEU-37.21-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/CUDA-12.x-76B900?style=for-the-badge&logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Dataset-Multi30k+opus100-blue?style=for-the-badge" />
</p>

A Neural Machine Translation system built **from scratch** in PyTorch, implementing the original **"Attention Is All You Need"** (Vaswani et al., 2017) architecture with modern training practices. No `nn.Transformer` used — every component is hand-coded.

---

## Stats

<table>
<tr>
<td>

**Model**
| | |
|---|---|
| Architecture | Transformer Base |
| Parameters | 52,295,680 |
| Encoder / Decoder | 6 layers each |
| Attention heads | 8 |
| d_model | 512 |
| d_ff | 2048 |
| Vocabulary | 16,000 (shared BPE) |

</td>
<td>

**Training**
| | |
|---|---|
| Dataset | Multi30k + opus100 (49k pairs) |
| Epochs | 45 |
| Batch size | 64 (accum × 4 = 256 eff.) |
| Optimizer | Adam |
| LR schedule | Noam (warmup=4000) |
| Peak LR | 7.0 × 10⁻⁴ |
| Hardware | RTX 3060 Laptop 6.4 GB |
| Time | ~8.5 hours |

</td>
<td>

**Results**
| | |
|---|---|
| Test BLEU | **37.21** |
| Val BLEU | **38.15** |
| Decoding | Greedy |
| Metric | sacrebleu 13a |
| Domain | Captions + web |

</td>
</tr>
</table>

---

## Training Curve

```
BLEU
 38 |                   ·····················
 35 |                ···
 30 |             ···
 25 |           ··
 20 |        ···
 15 |      ··
 10 |    ··
  5 |  ··
  0 |··
    +----+----+----+----+----> Epoch
    0    10   20   30   40 45

Val Loss
7.0 |·
5.5 | ·
4.5 |  ·
3.5 |   ·
2.9 |    ·····
2.7 |         ·················
    +----+----+----+----+----> Epoch
    0    10   20   30   40 45
```

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Mathematical Foundations](#2-mathematical-foundations)
3. [Project Structure](#3-project-structure)
4. [Installation](#4-installation)
5. [Reproducing Results](#5-reproducing-results)
6. [Hyperparameter Reference](#6-hyperparameter-reference)
7. [Results](#7-results)
8. [Implementation Notes](#8-implementation-notes)
9. [References](#9-references)

---

## 1. Architecture Overview

```
                          +----------------------------------+
         Source           |          ENCODER (x6)            |
    "The dog runs"        |                                  |
           |              |  +----------------------------+  |
           v              |  |  Multi-Head Self-Attention |  |
    +-------------+       |  |  (src_mask: PAD masking)   |  |
    |  Embedding  |       |  +------------+---------------+  |
    |  * sqrt(d)  |       |               | residual + LN    |
    +------+------+       |  +------------v---------------+  |
           |              |  | Position-Wise FFN (GELU)   |  |
    +------v------+       |  +------------+---------------+  |
    |  Positional |       |               | residual + LN    |
    |  Encoding   +------>+               v                  |
    +-------------+       |           memory Z               |
                          +---------------+------------------+
                                          |
                          +---------------v------------------+
         Target           |          DECODER (x6)            |
   "[SOS] Der Hund"       |                                  |
           |              |  +----------------------------+  |
           v              |  | Masked Multi-Head Attention |  |
    +-------------+       |  | (tgt_mask: causal + PAD)   |  |
    |  Embedding  |       |  +------------+---------------+  |
    |  * sqrt(d)  |       |               | residual + LN    |
    +------+------+       |  +------------v---------------+  |
           |              |  |  Cross-Attention           |  |
    +------v------+       |  |  Q=decoder, K=V=memory Z   |  |
    |  Positional +------>+  +------------+---------------+  |
    |  Encoding   |       |               | residual + LN    |
    +-------------+       |  +------------v---------------+  |
                          |  | Position-Wise FFN (GELU)   |  |
                          |  +------------+---------------+  |
                          |               | residual + LN    |
                          +---------------+------------------+
                                          |
                                   +------v------+
                                   |   Linear    |  (weight-tied to embedding)
                                   |  + Softmax  |
                                   +------+------+
                                          |
                                   "Der Hund rennt"
```

**Weight tying:** Source embedding, target embedding, and output projection all share the same weight matrix $W_E \in \mathbb{R}^{V \times d}$, reducing the parameter count by ~30M for a 16k vocabulary.

---

## 2. Mathematical Foundations

### 2.1 Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

The $\sqrt{d_k}$ scale factor prevents dot products from growing too large and pushing softmax into near-zero gradient regions. Masking is applied before softmax by filling masked positions with $-10^4$ (instead of $-\infty$) to avoid NaN gradients in fully-masked rows:

$$\text{scores} = \text{clamp}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M,\ \min=-10^4\right)$$

Two mask types are used:
- **PAD mask:** prevents attention to padding tokens in encoder self-attention and decoder cross-attention.
- **Causal mask:** upper-triangular matrix ensures position $i$ can only attend to positions $j \leq i$ in the decoder.

### 2.2 Multi-Head Attention

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\,W^O$$
$$\text{head}_i = \text{Attention}(QW_i^Q,\; KW_i^K,\; VW_i^V)$$

Implemented with a single fused projection $(d_{\text{model}}, 3 \cdot d_{\text{model}})$ split post-multiply — one large GEMM instead of $3h$ small ones, significantly more efficient on GPU Tensor Cores.

### 2.3 Positional Encoding

$$PE_{(pos,\, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \qquad PE_{(pos,\, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

Fixed (not learned), registered as a PyTorch buffer so it moves to GPU without appearing in the optimizer's parameter list.

### 2.4 Position-Wise Feed-Forward Network

$$\text{FFN}(x) = \text{GELU}(xW_1 + b_1)\,W_2 + b_2$$

GELU is used instead of the paper's ReLU — smoother gradient flow, standard in all modern language models (BERT, GPT series).

### 2.5 Pre-Layer Normalisation

Each sublayer uses Pre-LN residual connections (Xiong et al., 2020):

$$x \leftarrow x + \text{Dropout}\!\left(\text{Sublayer}\!\left(\text{LayerNorm}(x)\right)\right)$$

Pre-LN keeps gradient magnitudes stable at depth, unlike the original Post-LN which suffers exponential gradient decay at initialisation.

### 2.6 Noam Learning Rate Schedule

$$lr(\text{step}) = d_{\text{model}}^{-0.5} \cdot \min\!\left(\text{step}^{-0.5},\; \text{step} \cdot \text{warmup}^{-1.5}\right)$$

Peak at $\text{step} = \text{warmup} = 4000$:

$$lr_{\text{peak}} = \frac{1}{\sqrt{512 \times 4000}} \approx 7.0 \times 10^{-4}$$

**Critical implementation detail:** The scheduler must be stepped *before* the optimizer step. If called after, the first batch uses Adam's default $lr=1.0$, which catastrophically corrupts the model in a single update.

Adam optimizer: $\beta_1 = 0.9$, $\beta_2 = 0.98$, $\epsilon = 10^{-9}$ (paper values).

### 2.7 Label Smoothing

$$\tilde{y}_k = \begin{cases} 1 - \varepsilon & k = \text{correct class} \\ \varepsilon / (V - 2) & k \neq \text{correct class},\; k \neq \text{PAD} \\ 0 & k = \text{PAD} \end{cases}$$

With $\varepsilon = 0.1$. Loss is KL divergence between $\tilde{y}$ and model output, averaged over non-PAD tokens only. The loss must be computed in FP32 (not AMP FP16) due to numerical precision requirements across a 16k-dimensional vocabulary.

### 2.8 Beam Search with Length Penalty

Maintains $B$ partial hypotheses and ranks completed sequences using Wu et al. (2016) length penalty:

$$\text{score}_{\text{final}}(Y) = \frac{\sum_t \log P(y_t \mid y_{<t}, x)}{\left(\frac{5 + |Y|}{6}\right)^\alpha}$$

$\alpha = 0.6$ (Google NMT default). Length penalty is applied only at final ranking, not during beam pruning.

---

## 3. Project Structure

```
machine_translation/
|
+-- src/
|   +-- model.py        Transformer architecture (MHA, PE, FFN, Encoder, Decoder)
|   +-- tokenizer.py    BPE tokenizer training and loading (shared EN+DE vocab)
|   +-- dataset.py      Multi30k download, TranslationDataset, collate_fn, DataLoaders
|   +-- train.py        AMP training loop, Noam scheduler, label smoothing, checkpointing
|   +-- translate.py    Beam search inference, interactive REPL, file translation
|   +-- eval.py         Standalone test-set BLEU evaluation on best checkpoint
|
+-- checkpoints/        best.pt + epoch_NNN.pt snapshots
+-- data/
|   +-- raw/            HuggingFace dataset cache + combined_corpus.txt
|   +-- tokenizer/      tokenizer.json (trained BPE model)
+-- README.md
```

---

## 4. Installation

**Requirements:** Python >= 3.10, CUDA 12.x, NVIDIA GPU.

```bash
# 1. Navigate to project
cd machine_translation

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# 3. Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install remaining dependencies
pip install datasets tokenizers sacrebleu

# 5. Verify GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

**Windows note:** On Windows, `from datasets import ...` must be imported before any PyTorch CUDA initialization to avoid a segfault caused by a conflict between the datasets C extensions and the CUDA runtime. This is already handled in all source files.

---

## 5. Reproducing Results

### Step 1 — Training

Data download, tokenizer training, and model training all happen in one command:

```bash
# RTX 3060 Laptop (6 GB VRAM) or similar
python src/train.py --batch-size 64 --num-workers 0 --epochs 150 --accum-steps 4 --retrain-tokenizer --extra-dataset opus100 --extra-max 20000
```

The first run downloads Multi30k + opus100 (~65 MB total) and trains a shared BPE tokenizer (vocab=16k) automatically. Training converges around epoch 45; use Ctrl+C to stop early — best checkpoint is saved automatically.

**Resume from checkpoint:**
```bash
python src/train.py --batch-size 64 --num-workers 0 --epochs 150 --resume checkpoints/best.pt
```

### Step 2 — Evaluate on test set

```bash
python src/eval.py
```

Reports BLEU on the full test set using the best saved checkpoint with greedy decoding.

### Step 3 — Interactive translation

```bash
python src/translate.py --checkpoint checkpoints/best.pt --interactive
```

```
EN->DE Interactive Translator  (beam_width=5, alpha=0.6)

EN > Two dogs are playing in the park.
DE > zwei hunde spielen im park .  (95ms)

EN > A woman is reading a book.
DE > eine frau liest ein buch .  (112ms)
```

**Domain note:** The model is trained on Multi30k, which consists entirely of image captions. It translates visual scene descriptions well but will produce nonsense for conversational input ("hello", "I love you", etc.) since those patterns do not appear in the training data.

### Step 4 — Translate a single sentence

```bash
python src/translate.py --checkpoint checkpoints/best.pt \
    --sentence "The children are swimming in the lake."
```

---

## 6. Hyperparameter Reference

| Parameter | Value used | Description |
|---|---|---|
| `--d-model` | 512 | Embedding / hidden dimensionality |
| `--n-heads` | 8 | Attention heads per layer |
| `--n-layers` | 6 | Encoder and decoder depth |
| `--d-ff` | 2048 | FFN inner layer size |
| `--dropout` | 0.1 | Dropout probability (0 at inference) |
| `--vocab-size` | 16000 | Shared BPE vocabulary size |
| `--batch-size` | 64 | Sentence pairs per batch |
| `--epochs` | 100 | Training epochs |
| `--warmup-steps` | 4000 | Noam schedule linear warmup |
| `--label-smoothing` | 0.1 | Label smoothing epsilon |
| `--clip-grad` | 1.0 | Max gradient L2 norm |
| `--beam-width` | 5 | Beam search width at inference |
| `--length-penalty` | 0.6 | Wu et al. alpha exponent |

---

## 7. Results

**Hardware:** NVIDIA GeForce RTX 3060 Laptop GPU (6.4 GB VRAM)  
**Training time:** ~45 epochs, ~690 seconds/epoch, ~8.5 hours total  
**Parameters:** 52,295,680

| Metric | Value |
|---|---|
| Test BLEU (greedy) | **37.21** |
| Best val BLEU | **38.15** |

**Actual training curve (selected epochs):**

| Epoch | Train Loss | Val Loss | Val BLEU | Note |
|---|---|---|---|---|
| 1 | ~7.0 | ~5.5 | — | Warmup phase |
| 10 | ~3.5 | ~3.3 | ~18 | LR still rising |
| 18 | 2.83 | 2.71 | 38.12 | First best checkpoint |
| 34 | ~2.70 | ~2.68 | 38.15 | Best checkpoint |
| 45 | ~2.65 | ~2.72 | ~37.8 | End of training |

> BLEU is measured with sacrebleu `tokenize="13a"`. Val BLEU during training used greedy decoding for speed; the final eval.py result reflects the same.

**Sample translations:**

```
EN  Two dogs are playing in the park.
DE  zwei hunde spielen im park .

EN  A woman is reading a book.
DE  eine frau liest ein buch .

EN  The child is running on the street.
DE  das kind läuft auf der straße .

EN  A man in a red jacket is climbing a mountain.
DE  ein mann in einer roten jacke klettert einen berg .
```

> German umlauts (ä, ö, ü, ß) are preserved — `StripAccents` normalization is not applied. BPE subwords are joined with spaces during decoding, so punctuation appears separated (e.g., `ende .`); both hypothesis and reference go through the same path, keeping BLEU internally consistent.

**Comparison with baselines:**

| System | BLEU |
|---|:---:|
| RNN seq2seq (Luong, 2015) | ~27 |
| This repo — Multi30k only | 35.60 |
| This repo — **+opus100 (49k pairs)** | **37.21** |
| State-of-the-art (large ensembles) | ~40+ |

---

## 8. Implementation Notes

### Attention score clamping

Padding-only decoder positions produce rows of all $-\infty$ in the attention score matrix, causing $0/0 = \text{NaN}$ in the softmax. Using `nan_to_num` fixes the forward value but leaves NaN in the autograd graph, corrupting gradients. Clamping to $-10^4$ before softmax prevents NaN entirely:

```python
scores = scores.clamp(min=-1e4)
attn_weights = F.softmax(scores, dim=-1)
```

### Label smoothing in FP32

Under `torch.autocast`, the loss forward pass runs in FP16. With a vocabulary of 16k tokens, summing 16k FP16 log-probabilities causes precision loss. The loss module explicitly casts logits to FP32:

```python
logits = logits.float()  # opt out of AMP for this operation
```

### Windows-specific import ordering

Loading HuggingFace `datasets` after CUDA is initialized causes a segfault on Windows due to a conflict between the datasets C extensions and the CUDA runtime. All entry-point files import `datasets` before any `torch` import.

### Decoder token spacing

The BPE tokenizer uses a Whitespace pre-tokenizer without end-of-word suffixes. The default `BPEDecoder` concatenates all tokens without spaces. Decoding is instead done by joining token strings with spaces:

```python
tokens = [self._tok.id_to_token(id) for id in ids if id not in special_ids]
return " ".join(tokens)
```

Both hypothesis and reference go through the same path, so BLEU comparison remains internally consistent.

---

## 9. References

- Vaswani, A. et al. (2017). **Attention Is All You Need.** *NeurIPS 2017.* [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Sennrich, R. et al. (2016). **Neural Machine Translation of Rare Words with Subword Units.** *ACL 2016.* [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)
- Hendrycks, D. & Gimpel, K. (2016). **Gaussian Error Linear Units (GELUs).** [arXiv:1606.08415](https://arxiv.org/abs/1606.08415)
- Szegedy, C. et al. (2016). **Rethinking the Inception Architecture.** *CVPR 2016.*
- Ba, J. et al. (2016). **Layer Normalization.** [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)
- Press, O. & Wolf, L. (2017). **Using the Output Embedding to Improve Language Models.** *EACL 2017.* [arXiv:1608.05859](https://arxiv.org/abs/1608.05859)
- Wu, Y. et al. (2016). **Google's Neural Machine Translation System.** [arXiv:1609.08144](https://arxiv.org/abs/1609.08144)
- Xiong, R. et al. (2020). **On Layer Normalization in the Transformer Architecture.** *ICML 2020.* [arXiv:2002.04745](https://arxiv.org/abs/2002.04745)
- Post, M. (2018). **A Call for Clarity in Reporting BLEU Scores.** *WMT 2018.* (sacrebleu)
