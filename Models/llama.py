import torch.nn as nn
import torch


class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6):
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
        inv_freq = torch.tensor(
            [1 / theta ** (i / self.head_dim) for i in range(0, self.head_dim, 2)]
        )
        angles = torch.outer(positions, inv_freq)

        self.register_buffer(
            "cos", torch.cos(angles)
        )  # [context_lenght , head_dim / 2]
        self.register_buffer(
            "sin", torch.sin(angles)
        )  # [context_lenght , head_dim / 2]

    def forward(self, x):
        x_even = x[..., 0::2]  # [batch, head_dim, n_tokens, emb_dim / 2]
        x_odd = x[..., 1::2]  # [batch, head_dim, n_tokens, emb_dim / 2]

        even_rot = x_even * self.cos - x_odd * self.sin
        odd_rot = x_even * self.sin + x_odd * self.cos

        return torch.tensor((even_root, odd_root))


class GroupedQueryAttention(nn.Module):
    def __init__(self, context_length, emb_dim, num_heads, num_kv_heads, dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        assert emb_dim % num_heads == 0
        assert num_heads % num_kv_heads == 0

        self.head_dim = emb_dim // num_heads

        self.q_project = nn.Linear(
            self.emb_dim, self.num_heads * self.head_dim, bias=False
        )
        self.k_project = nn.Linear(
            self.emb_dim, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_project = nn.Linear(
            self.emb_dim, self.num_kv_heads * self.head_dim, bias=False
        )
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )  # Register mask in buffer for helping computation

        self.out_project = nn.Linear(self.emb_dim, self.emb_dim, bias=False)

    def forward(self, x):
        batches, n_tokens, _ = x.shape
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

        k = k.repeat_interleave(
            1, self.num_heads // self.num_kv_heads, 1, 1
        )  # Repeats to share K with multiple Q
        v = v.repeat_interleave(
            1, self.num_heads // self.num_kv_heads, 1, 1
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


class DecoderBlock(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.RMSNorm_1 = RMSNorm(emb_dim)
        self.RMSNorm_2 = RMSNorm(emb_dim)
        self.group_query_attention_blocks = GroupedQueryAttention()
        self.FeedForward = "pending"

    def forward(self, x):
        pass


class OutputBlock(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.RMSNorm = RMSNorm(emb_dim)
        self.linear_projection = "pending"


class Llama(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config["vocab_size"], config["embed_dim"])
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock() for _ in range(config["n_layers"])]
        )
        self.output_block = OutputBlock()

    def forward(self, x):
        pass
