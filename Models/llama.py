import torch.nn as nn
import torch


class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x):  # x [batch_size , seq_len , emb_dim]
        squared_mean = torch.pow(x, 2).mean(
            dim=-1, keepdim=True
        )  # [batch_size, seq_len , 1]
        rms = torch.rsqrt(self.eps + squared_mean)
        return x * rms * self.gamma


class RotaryPositionEncoding(nn.Module):
    def __init__(self, context_length, head_dim, theta=10000.0):
        super().__init__()
        self.context_length = context_length
        self.head_dim = head_dim

        assert head_dim % 2 == 0
        positions = torch.arange(self.context_length)

        idx = torch.arange(0, head_dim, 2).float()
        inv_freq = 1.0 / (theta ** (idx / head_dim))
        angles = torch.outer(positions, inv_freq)

        self.register_buffer(
            "cos", torch.cos(angles)[None, None, :, :]
        )  # [1,1,context_lenght , head_dim / 2]
        self.register_buffer(
            "sin", torch.sin(angles)[None, None, :, :]
        )  # [1,1,context_lenght , head_dim / 2]

    def forward(self, x):
        _, _, seq_length, _ = x.shape

        if seq_length > self.context_length:
            raise ValueError("sequence length exceeds context length")

        x_even = x[..., 0::2]  # [batch, heads, seq_lenght, head_dim / 2]
        x_odd = x[..., 1::2]  # [batch, heads, seq_lenght, head_dim / 2]

        even_rot = (
            x_even * self.cos[:, :, :seq_length, :]
            - x_odd * self.sin[:, :, :seq_length, :]
        )
        odd_rot = (
            x_even * self.sin[:, :, :seq_length, :]
            + x_odd * self.cos[:, :, :seq_length, :]
        )

        encoding = torch.empty_like(x)
        encoding[..., 0::2] = even_rot
        encoding[..., 1::2] = odd_rot

        return encoding


class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_rope = config["use_rope"]
        self.emb_dim = config["emb_dim"]
        self.num_heads = config["num_heads"]
        self.num_kv_heads = config["num_kv_heads"]
        self.context_length = config["context_length"]

        assert self.emb_dim % self.num_heads == 0
        assert self.num_heads % self.num_kv_heads == 0

        self.head_dim = self.emb_dim // self.num_heads

        self.q_project = nn.Linear(
            self.emb_dim, self.num_heads * self.head_dim, bias=False
        )
        self.k_project = nn.Linear(
            self.emb_dim, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_project = nn.Linear(
            self.emb_dim, self.num_kv_heads * self.head_dim, bias=False
        )
        if self.use_rope:
            self.rotary_positional_encoding = RotaryPositionEncoding(
                self.context_length, self.head_dim
            )
        self.dropout = nn.Dropout(config["dropout"])
        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(self.context_length, self.context_length), diagonal=1
            ),
        )  # Register mask in buffer for helping computation

        self.out_project = nn.Linear(self.emb_dim, self.emb_dim, bias=False)

    def forward(self, x):
        batches, n_tokens, _ = x.shape

        if n_tokens > self.context_length:
            raise ValueError("sequence length exceeds context length")

        q = self.q_project(x)
        k = self.k_project(x)
        v = self.v_project(x)

        # Split representation into different heads. Each head works on a different slice of the Q/K/V projection.
        q = q.view(batches, n_tokens, self.num_heads, self.head_dim)
        k = k.view(batches, n_tokens, self.num_kv_heads, self.head_dim)
        v = v.view(batches, n_tokens, self.num_kv_heads, self.head_dim)

        q = q.permute(0, 2, 1, 3)  # batch, head, token, head_dimension
        k = k.permute(0, 2, 1, 3)  # batch, head, token, head_dimension
        v = v.permute(0, 2, 1, 3)  # batch, head, token, head_dimension

        if self.use_rope:
            q = self.rotary_positional_encoding(q)
            k = self.rotary_positional_encoding(k)

        k = k.repeat_interleave(
            self.num_heads // self.num_kv_heads, dim=1
        )  # Repeats to share K with multiple Q
        v = v.repeat_interleave(
            self.num_heads // self.num_kv_heads, dim=1
        )  # Repeats to share v with multiple Q

        attn_scores = (
            q @ k.transpose(2, 3)
        )  # (batch, head, tokens, head_dim) @ (batch, head, head_dim, tokens) = (batch, head, tokens, tokens)

        attn_scores = attn_scores * self.head_dim**-0.5  # Scaling

        mask_bool = self.mask.bool()[:n_tokens, :n_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vector = attn_weights @ v
        context_vector = context_vector.transpose(1, 2)

        context_vec = context_vector.contiguous().view(
            batches, n_tokens, self.emb_dim
        )  # Reshapes context vector (batch, tokens, d_out)

        # Join all heads info using a projection layer
        context_vec = self.out_project(context_vec)

        return context_vec


class FeedForward(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.hidden_dimension = int(2 / 3 * 4 * emb_dim)
        self.gate_proj = nn.Linear(emb_dim, self.hidden_dimension, bias=False)
        self.up_proj = nn.Linear(emb_dim, self.hidden_dimension, bias=False)
        self.silu = nn.SiLU()
        self.down_proj = nn.Linear(self.hidden_dimension, emb_dim, bias=False)

    def forward(self, x):
        gate = self.silu(self.gate_proj(x))
        up = self.up_proj(x)

        x = gate * up

        return self.down_proj(x)


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.RMSNorm_1 = RMSNorm(config["emb_dim"])
        self.RMSNorm_2 = RMSNorm(config["emb_dim"])
        self.group_query_attention = GroupedQueryAttention(config)
        self.feed_forward = FeedForward(config["emb_dim"])

    def forward(self, x):
        x_ = x
        x = self.RMSNorm_1(x)
        x = self.group_query_attention(x)

        x = x + x_
        x_ = x

        x = self.RMSNorm_2(x)
        x = self.feed_forward(x)

        return x + x_


class OutputBlock(nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        self.RMSNorm = RMSNorm(config["emb_dim"])
        self.linear_projection = nn.Linear(
            config["emb_dim"], config["vocab_size"], bias=False
        )

        self.linear_projection.weight = weights

    def forward(self, x):
        x = self.RMSNorm(x)
        logits = self.linear_projection(x)

        return logits


class Llama(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config["n_layers"])]
        )
        self.output_block = OutputBlock(config, self.token_embedding.weight)

    def forward(self, x):
        x = self.token_embedding(x)
        for block in self.decoder_blocks:
            x = block(x)
        x = self.output_block(x)

        return x
