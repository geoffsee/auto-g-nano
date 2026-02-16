# auto-g-nano

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-orange)](https://huggingface.co/models?search=auto-g-nano)

~~~
(auto-g-nano) ➜  auto-g-nano git:(auto-g-nano-2) uv run python3 generate.py                   
Loading FineWeb-Edu (train)...
Resolving data files: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2410/2410 [00:00<00:00, 94196.05it/s]
Model created with 1.06B params
Successfully loaded model.pt
------------------------------
In the future, artificial intelligence will scatter the light, making the images too noisy to be useful. Discovering almost 1,000 comets since SOHO's launch on December 2, 1995 is a testament to the skill of the LASCO team."
SOHO successfully completed its primary mission in April 1998. It has enough fuel to remain on station to keep hunting comets for decades if the LASCO continues to function.
For information about SOHO on the Internet, visit:
Explore further: Long
------------------------------
(auto-g-nano) ➜  auto-g-nano git:(auto-g-nano-2) 
~~~


A complete, minimal, and fully functional decoder-only Language Model (nanoGPT-style) built from absolute scratch in pure PyTorch. No high-level abstractions, no pre-built modules—just the essentials of the Transformer architecture.

---

## 📖 Table of Contents
- [Features](#-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Training](#-training)
- [Inference](#-inference)
- [Hugging Face Integration](#-hugging-face-integration)
- [Results](#-results)
- [Acknowledgments](#-acknowledgments)
- [License](#-license)

---

## ✨ Features
- **Pure PyTorch**: Implemented using basic `nn.Module` and `functional` operations.
- **NanoGPT Style**: Focused on simplicity and readability, ideal for learning.
- **Character-level Tokenization**: Simple and effective for small datasets like Tiny Shakespeare.
- **Hugging Face Hub Support**: Built-in scripts to push and pull models from the Hub.
- **Modern Tooling**: Managed with `uv` for fast, reproducible environments.

## 🏗 Architecture
The model is a modernized Transformer (Nemotron-style) with:
- **Token Embeddings**: Untied embeddings for maximum capacity.
- **RoPE (Rotary Positional Embeddings)**: Using $\theta=500,000$ for better long-context handling.
- **Grouped-Query Attention (GQA)**: Efficient attention mechanism (32 Q heads, 8 KV heads).
- **Nemotron-style FFN**: SwiGLU activation with a 4x hidden dimension expansion.
- **RMSNorm**: Applied before each sub-layer (Pre-Norm).
- **Residual Connections**: Stable deep network architecture.
- **GPT-2 Style Initialization**: Refined weight initialization with $1/\sqrt{2 \times layers}$ scaling for residual layers.
- **FlashAttention support**: Uses `scaled_dot_product_attention` for high-performance training and inference.

| Hyperparameter | Value |
| :--- | :--- |
| Parameters | ~1.06B |
| Context Length | 1024 |
| Embedding Dim | 2048 |
| Heads | 32 |
| KV Heads | 8 |
| Layers | 14 |
| Dropout | 0.0 |
| Precision | Mixed (bfloat16/float16) |
| Optimizer | AdamW + Cosine Decay |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended)

### Installation
```bash
git clone https://github.com/your-username/auto-g-nano.git
cd auto-g-nano
uv sync
```

### Download Dataset
Fetch the Tiny Shakespeare dataset:
```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

## 🧠 Instruction Fine-Tuning
The model supports Supervised Fine-Tuning (SFT) using the ChatML format.

### Prompt Guidelines (ChatML)
For best results with the instruct model, use the ChatML template:
```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{your_question}<|im_end|>
<|im_start|>assistant
```

### How to Fine-Tune
To transition the base model into an assistant, run the fine-tuning script:
```bash
uv run python finetune.py
```
This uses the `SmolTalk` dataset and a lower learning rate to adapt the 1.06B parameters to follow instructions.

---

## 🏋️ Training & Fine-Tuning
- **Pre-training**: `uv run python train.py` (Trains on FineWeb-Edu)
- **Fine-tuning**: `uv run python finetune.py` (Trains on SmolTalk Instruct)

---

## 🔮 Inference
Generate text or chat with the model:
```bash
uv run python generate.py
```
You can toggle `chat_mode = True` in `generate.py` to use the instruction-following template.

---

## 🤗 Hugging Face Integration

### Publishing
Upload your model and code to the Hugging Face Hub:
```bash
uv run python publish.py <your-username>/<your-model-name>
```

### Loading
Load a published model directly in your code:
```python
from model import GPT
model = GPT.from_pretrained("<your-username>/<your-model-name>")
```

---

## 📊 Results

### Training Curve
Expect the loss to start around ~4.3 and drop to ~1.5 after 5000 iterations.
```console
step    0 | train loss 4.2932 | val loss 4.2972
step 1000 | train loss 1.3553 | val loss 1.5904
step 2500 | train loss 1.0986 | val loss 1.4848
step 5000 | train loss 0.8442 | val loss 1.5871
```

### Sample Output
> **LADY GREY:**
> And their I know before I descend.
> 
> **KING EDWARD IV:**
> But what prophecy they love to do thee for thee?
> 
> **GLOUCESTER:**
> The bridge makes her majesty to be at once.

---

## 🙏 Acknowledgments
- Inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).
- Dataset provided by the Tiny Shakespeare corpus.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.