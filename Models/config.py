GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": True,  # Query-Key-Value bias
}

GPT_CONFIG_124M_LOW = {
    "vocab_size": 50257,
    "context_length": 64,
    "emb_dim": 256,
    "n_heads": 8,
    "n_layers": 6,
    "drop_rate": 0.2,
    "qkv_bias": False,
}

LLAMA_CONFIG = {
    "vocab_size": 32000,
    "context_length": 256,
    "emb_dim": 512,
    "n_layers": 8,
    "num_heads": 8,
    "num_kv_heads": 2,
    "dropout": 0.1,
    "use_rope": True,
}
