from dataclasses import dataclass
from math import log, sqrt

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    src_vocab: int
    tgt_vocab: int
    width: int = 128
    heads: int = 4
    layers: int = 2
    ff_width: int = 512
    dropout: float = 0.1
    positions: int = 128
    course_norm: bool = False

    def __post_init__(self):
        if min(self.src_vocab, self.tgt_vocab) < 5:
            raise ValueError('词表至少需要 5 个 token。')
        if self.heads < 1 or self.width < 2 or self.width % self.heads or self.width % 2:
            raise ValueError('隐藏维度必须是正偶数且能被头数整除。')
        if min(self.layers, self.ff_width) < 1 or self.positions < 2 or not 0 <= self.dropout < 1:
            raise ValueError('模型层数、前馈维度、序列长度或 dropout 无效。')


class TokenPositions(nn.Module):
    def __init__(self, vocab, config):
        super().__init__()
        self.tokens = nn.Embedding(vocab, config.width, padding_idx=0)
        self.dropout = nn.Dropout(config.dropout)
        self.scale = sqrt(config.width)
        frequencies = (-log(10000.0) * torch.arange(0, config.width, 2) / config.width).exp()
        phases = torch.arange(config.positions).unsqueeze(1) * frequencies
        self.register_buffer('position', torch.stack((phases.sin(), phases.cos()), dim=-1).flatten(1)[None])

    def forward(self, ids):
        length = ids.size(1)
        if length > self.position.size(1):
            raise ValueError(f'序列长度 {length} 超过模型的位置上限 {self.position.size(1)}。')
        return self.dropout(self.tokens(ids) * self.scale + self.position[:, :length])


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.head_width = config.width // config.heads
        self.query = nn.Linear(config.width, config.width)
        self.key = nn.Linear(config.width, config.width)
        self.value = nn.Linear(config.width, config.width)
        self.output = nn.Linear(config.width, config.width)
        self.dropout = nn.Dropout(config.dropout)

    def split_heads(self, x):
        return x.unflatten(-1, (self.heads, self.head_width)).transpose(1, 2)

    def forward(self, query, context, allowed):
        q = self.split_heads(self.query(query))
        k = self.split_heads(self.key(context))
        v = self.split_heads(self.value(context))
        scores = (q @ k.transpose(-2, -1)) / sqrt(self.head_width)
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        # 整行被遮住时，softmax 仍会返回均匀分布。
        weights = scores.softmax(dim=-1).masked_fill(~allowed, 0)
        mixed = (self.dropout(weights) @ v).transpose(1, 2).flatten(2)
        return self.output(mixed)


class CourseNorm(nn.Module):
    """旧权重的归一化使用样本方差，保留该公式用于兼容。"""

    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.bias = nn.Parameter(torch.zeros(width))

    def forward(self, x):
        variance, mean = torch.var_mean(x, dim=-1, correction=1, keepdim=True)
        return (x - mean) * (variance + 1e-6).rsqrt() * self.weight + self.bias


def normalization(config):
    return CourseNorm(config.width) if config.course_norm else nn.LayerNorm(config.width, eps=1e-6)


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention = Attention(config)
        self.attention_norm = normalization(config)
        self.ff_norm = normalization(config)
        self.ff = nn.Sequential(nn.Linear(config.width, config.ff_width), nn.ReLU(),
                                nn.Dropout(config.dropout), nn.Linear(config.ff_width, config.width))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, allowed):
        normalized = self.attention_norm(x)
        x = x + self.dropout(self.self_attention(normalized, normalized, allowed))
        return x + self.dropout(self.ff(self.ff_norm(x)))


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention = Attention(config)
        self.cross_attention = Attention(config)
        self.attention_norm = normalization(config)
        self.cross_norm = normalization(config)
        self.ff_norm = normalization(config)
        self.ff = nn.Sequential(nn.Linear(config.width, config.ff_width), nn.ReLU(),
                                nn.Dropout(config.dropout), nn.Linear(config.ff_width, config.width))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, memory, source_allowed, target_allowed):
        normalized = self.attention_norm(x)
        x = x + self.dropout(self.self_attention(normalized, normalized, target_allowed))
        x = x + self.dropout(self.cross_attention(self.cross_norm(x), memory, source_allowed))
        return x + self.dropout(self.ff(self.ff_norm(x)))


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.source = TokenPositions(config.src_vocab, config)
        self.target = TokenPositions(config.tgt_vocab, config)
        self.encoder = nn.ModuleList(EncoderBlock(config) for _ in range(config.layers))
        self.decoder = nn.ModuleList(DecoderBlock(config) for _ in range(config.layers))
        self.encoder_norm = normalization(config)
        self.decoder_norm = normalization(config)
        self.projection = nn.Linear(config.width, config.tgt_vocab)
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)
        with torch.no_grad():
            self.source.tokens.weight[0].zero_()
            self.target.tokens.weight[0].zero_()

    def encode(self, source_ids):
        source_allowed = source_ids.ne(0)[:, None, None, :]
        x = self.source(source_ids)
        for layer in self.encoder:
            x = layer(x, source_allowed)
        return self.encoder_norm(x), source_allowed

    def decode(self, target_ids, memory, source_allowed):
        length = target_ids.size(1)
        causal = torch.ones(length, length, dtype=torch.bool, device=target_ids.device).tril()
        target_allowed = causal & target_ids.ne(0)[:, None, None, :]
        x = self.target(target_ids)
        for layer in self.decoder:
            x = layer(x, memory, source_allowed, target_allowed)
        return self.decoder_norm(x)

    def forward(self, source_ids, target_ids):
        memory, allowed = self.encode(source_ids)
        return self.projection(self.decode(target_ids, memory, allowed))
