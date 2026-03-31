# LLM from Scratch 🧠

A ground-up implementation of a GPT-2-style Large Language Model in PyTorch, built entirely from scratch no HuggingFace Trainer, no magic abstractions. The notebook covers the full pipeline: architecture, pretraining, weight transfer from OpenAI's GPT-2, and fine-tuning for downstream classification.

---

## Overview

This project was built for deep educational understanding of how LLMs actually work under the hood. Rather than calling `AutoModel.from_pretrained()` and calling it a day, every component from multi-head attention to the training loop is implemented manually.

The notebook is split into two main sections:

### 1. 📖 Language Modeling (Pretraining)
Training a GPT-2-scale model to predict the next token on a raw text corpus.

- Custom `GPTDataset` and `CreateDataLoader` with sliding window tokenization
- `MultiHeadAttention` with configurable heads, dropout, and QKV bias
- Full `GptModel` with token + positional embeddings, stacked transformer blocks, and layer norm
- `GenerateText` with **top-k sampling** and **temperature scaling**
- `Trainer` class with a full train/validation loop using AdamW + weight decay
- **Weight transfer** utility (`TransferWeights`) to load pretrained GPT-2 weights from HuggingFace into the custom architecture — verifying shape compatibility at every layer

### 2. 📧 Text Classification (Fine-tuning)
Adapting the pretrained GPT-2 backbone for **spam detection** on an email dataset.

- EDA: target distribution, sentence length, word count, mean word length — grouped by class
- Fine-tuning with only the classification head unfrozen (last linear layer)
- `ClassificationTrainer` with **Stratified K-Fold cross-validation** and per-fold metrics
- Experiment comparing training **with vs. without gradient clipping** to handle exploding gradients
- Training curve visualization: accuracy, loss across folds

---

## Architecture

| Component | Details |
|---|---|
| Vocab size | 50,257 (GPT-2 BPE via `tiktoken`) |
| Model dimension | 768 |
| Attention heads | 12 |
| Transformer blocks | 12 |
| Max sequence length | 1,024 |
| Tokenizer | `tiktoken` (GPT-2 encoding) |

Config is centralized in a `Config` class for easy experimentation.

---

## Key Components

```
GptModel
├── Token Embeddings          (nn.Embedding)
├── Positional Embeddings     (nn.Embedding)
├── Embedding Dropout
├── Transformer Blocks × 12
│   ├── LayerNorm
│   ├── MultiHeadAttention    (causal, with dropout)
│   ├── LayerNorm
│   └── FeedForward MLP
└── Output Head               (linear → vocab logits)
```

---

## Experiments

### Weight Transfer
Verified that the custom `GptModel` can load official GPT-2 weights from HuggingFace (`GPT2Model.from_pretrained('gpt2')`), with shape-matching assertions at every parameter group (embeddings, attention projections, MLP layers, layer norms).

### Gradient Explosion
Demonstrates the effect of gradient clipping on training stability for the classification task — comparing loss/accuracy curves with and without `torch.nn.utils.clip_grad_norm_`.

---

## Setup

```bash
pip install torch tiktoken datasets scikit-learn transformers
```

The pretraining notebook expects a `.txt` file at `/kaggle/input/text-data/`.  
The classification section uses the [Email Spam Detection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).

---

## Usage

```python
# Generate text (after training or weight transfer)
input_txt = "I HAD always thought Jack Gisburn rather"
tkn_id = torch.tensor(tokenizer.encode(input_txt)).unsqueeze(0).to("cuda:0")

output = GenerateText(
    model=gpt_model,
    token_ids=tkn_id,
    max_new_tokens=50,
    context_size=50,
    tokenizer=tokenizer,
    top_k=50,
    temperature=1.0
)
```

---

## Results

| Experiment | Notes |
|---|---|
| Pretraining | Train/val loss tracked over 100 epochs |
| Weight Transfer | Custom model output matches HuggingFace GPT-2 after transfer |
| Spam Classification (no clipping) | Baseline — susceptible to gradient explosion |
| Spam Classification (with clipping) | Stable training, consistent cross-fold accuracy |

---

## Motivation
 -> *"I don't just want to use these models — I want to understand every line."*

---

## License

MIT
