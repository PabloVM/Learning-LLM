# Learning LLMs — From Fundamentals to Practical Implementations

This repository documents my hands-on journey learning and implementing Large Language Models from the ground up using PyTorch.

The goal is not to build another high-level tutorial repository, but to deeply understand how modern LLM systems work internally by implementing core components manually, experimenting with training pipelines, and reproducing key concepts step by step.

The repository progressively evolves from low-level preprocessing and dataset creation to complete transformer architectures, pretrained weight loading, and downstream fine-tuning tasks.

It serves both as a personal learning log and as a practical showcase of applied understanding of transformer-based architectures and LLM workflows.

---

## Current Features

### Data preprocessing pipeline

- Text preprocessing and token preparation
- Vocabulary/token handling
- Sequence generation
- Training sample creation
- Dataset and DataLoader preparation using PyTorch

### GPT-2 implementation from scratch

A decoder-only transformer implementation inspired by GPT-2 architecture, including:

- Token embeddings
- Positional embeddings
- Multi-head self-attention
- Causal masking
- Feed-forward networks
- Residual connections
- Layer normalization
- Transformer blocks
- Autoregressive text generation

[GPT2 implementation](Models/gpt2.py)

Implemented directly in PyTorch without relying on high-level transformer abstractions.

### LLama-2 implementation from scratch

A decoder-only transformer implementation inspired by LLAMA-2 architecture, including:

- Token embeddings
- Rotary position encoding
- Multi-head self-attention
- Causal masking
- Feed-forward networks
- Residual connections
- Layer normalization
- Transformer blocks
- Autoregressive text generation

[LLAMA2 implementation](Models/llama.py)

Implemented directly in PyTorch without relying on high-level transformer abstractions.

### Pretrained GPT-2 weight loading

- Loading pretrained GPT-2 weights into the custom implementation
- Parameter mapping between architectures
- Validation of compatibility and inference behavior

### Post-training / downstream fine-tuning

- Sentiment classification task using the GPT backbone
- Adaptation of transformer outputs for classification
- Training loop and evaluation workflow
- Fine-tuning experiments and inference testing

---

## Technical Focus

This repository focuses on understanding the mechanics behind LLM systems rather than only using existing high-level frameworks.

Areas explored include:

- Transformer internals
- Attention mechanisms
- Tokenization workflows
- Training dynamics
- Weight initialization
- Autoregressive generation
- Fine-tuning strategies
- PyTorch implementation patterns
- Model interoperability with pretrained checkpoints

The emphasis is on readability, experimentation, and architectural understanding.

---

## Why This Repository Exists

Modern AI tooling increasingly abstracts away the underlying systems behind LLMs.

While these abstractions are extremely useful in production, they can also hide many of the engineering and mathematical principles that make these models work.

This repository is my way of:

- rebuilding those concepts from first principles,
- validating understanding through implementation,
- and developing practical intuition around transformer architectures and training pipelines.

---

## Planned Additions

Some future areas I plan to explore include:

- Training small language models from scratch
- BPE tokenizer implementation
- Rotary positional embeddings (RoPE)
- KV caching
- Flash Attention
- LoRA / PEFT fine-tuning
- Quantization
- RAG pipelines
- Mixture of Experts (MoE)
- RLHF / preference optimization
- Inference optimization
- Distributed training experiments

---

## Tech Stack

- Python
- PyTorch
- NumPy
- Jupyter Notebooks

---

## Philosophy

The objective is not just to use LLMs, but to understand them deeply enough to build, modify, debug, and adapt them to real-world systems.

Most implementations in this repository prioritize:

- clarity over abstraction,
- transparency over convenience,
- and learning over framework dependency.

---

## Disclaimer

This repository is an active learning project and is continuously evolving.

Some implementations intentionally prioritize educational clarity over production-level optimization.

---

## Useful resources

- Implementing LLMS from scratch : https://github.com/rasbt/LLMs-from-scratch

## Author

Developed by Pablo Valero

GitHub Repository:
https://github.com/PabloVM/Learning-LLM
