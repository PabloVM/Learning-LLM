import torch
import torch.nn as nn

# Implemetation based on the book "Build a Large Language Model(From Scratch)"


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # Epsilon is added to the variant to prevent division by 0 during normalization
        # Added scale and shift allow the model to recover useful ditribuitions after normalization.
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)  # Keep dimension preserves tensor dimension
        var = x.var(
            dim=-1, keepdim=True, unbiased=False
        )  # We set unbiased to false because we have the full tensor
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(
                        torch.tensor(2.0 / torch.pi) * (x + 0.044715 * torch.pow(x, 3))
                    )
                )
            )
        )


class FeedForward(nn.Module):
    """
    FeedForward is a small neural network consisting of two linear layers joined by a GELU activation function.
    This layer expands the dimension of the tensors 4 times and then reduces back to original size.
    This helps the network learn richer token representations.

    """

    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config["emb_dim"], 4 * config["emb_dim"]),
            GELU(),
            nn.Linear(4 * config["emb_dim"], config["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_lenght, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out  # Output dimension
        self.num_heads = num_heads  # Number of heads
        self.head_dim = d_out // num_heads  # Dimension of each head

        # QKV
        self.Q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.K = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.V = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.projection = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(
            dropout
        )  # Dropout regularizes attention weights during training
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_lenght, context_lenght), diagonal=1)
        )  # Register mask in buffer for helping computation

    def forward(self, x):
        batches, n_tokens, d_in = x.shape

        # Calculate qkv
        queries = self.Q(x)  # (batches, n_tokens, d_out)
        keys = self.K(x)  # (batches, n_tokens, d_out)
        values = self.V(x)  # (batches, n_tokens, d_out)

        # Split representation into different heads. Each head works on a different slice of the Q/K/V projection.

        queries = queries.view(
            batches, n_tokens, self.num_heads, self.head_dim
        )  # batch, token, head, head_dimension
        keys = keys.view(
            batches, n_tokens, self.num_heads, self.head_dim
        )  # batch, token, head, head_dimension
        values = values.view(
            batches, n_tokens, self.num_heads, self.head_dim
        )  # batch, token, head, head_dimension

        # Rearrange the tensor to computate attention per head

        queries = queries.permute(0, 2, 1, 3)  # batch, head, token, head_dimension
        keys = keys.permute(0, 2, 1, 3)  # batch, head, token, head_dimension
        values = values.permute(0, 2, 1, 3)  # batch, head, token, head_dimension

        attn_scores = (
            queries @ keys.transpose(2, 3)
        )  # (batch, head, tokens, head_dim) @ (batch, head, head_dim, tokens) = (batch, head, tokens, tokens)

        # Apply mask to avoid tokens can look in future tokens

        mask_bool = self.mask.bool()[:n_tokens, :n_tokens]
        attn_scores.masked_fill_(
            mask_bool, -torch.inf
        )  # -torch.inf becomes 0 when applying softmax

        # Applying softmax attention scores turn into attention weights
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Create context vector multipliying attention and values
        context_vec = attn_weights @ values  # (batch, heads, tokens, head_dim)
        context_vec = context_vec.transpose(1, 2)  # ( batch, tokens, heads, head_dim)

        context_vec = context_vec.contiguous().view(
            batches, n_tokens, self.d_out
        )  # Reshapes context vector (batch, tokens, d_out)

        # Join all heads info using a projection layer
        context_vec = self.projection(context_vec)

        return context_vec


class TransformerBlock(nn.Module):
    """Full GPT2 Transformer block implementation"""

    def __init__(self, config):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=config["emb_dim"],
            d_out=config["emb_dim"],
            context_lenght=config["context_length"],
            num_heads=config["n_heads"],
            dropout=config["drop_rate"],
            qkv_bias=config["qkv_bias"],
        )
        self.feedforward = FeedForward(config)
        self.norm1 = LayerNorm(config["emb_dim"])
        self.norm2 = LayerNorm(config["emb_dim"])
        self.dropout = nn.Dropout(config["drop_rate"])

    def forward(self, x):
        # First sub-block: LayerNorm -> Multi-head attention -> Dropout -> Residual connection
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.dropout(x)
        x = x + shortcut
        # Second sub-block: LayerNorm -> FeedForward -> Dropout -> Residual connection
        shortcut = x
        x = self.norm2(x)
        x = self.feedforward(x)
        x = self.dropout(x)
        x = x + shortcut

        return x


class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Converts token ids into dense vectors (vocab_size -> emb_dim)
        self.tok_emb = nn.Embedding(config["vocab_size"], config["emb_dim"])
        # Learns a vector representation for each position do the model can understand order
        self.pos_emb = nn.Embedding(config["context_length"], config["emb_dim"])

        self.drop_emb = nn.Dropout(config["drop_rate"])
        # Stack of transformers blocks
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config["n_layers"])]
        )

        self.last_norm = LayerNorm(config["emb_dim"])
        # Maps embeddings back to vocabulary dimensions producing logits for next-token prediction
        self.out_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        token_embedding = self.tok_emb(in_idx)
        position_embedding = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        x = (
            token_embedding + position_embedding
        )  # Combine token meaning + positional information
        x = self.drop_emb(x)
        x = self.transformer_blocks(x)
        x = self.last_norm(x)

        logits = self.out_head(x)

        return logits
