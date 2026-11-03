from dataclasses import asdict
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

import torch

from .model import ModelConfig, Transformer


def save_checkpoint(path, model, tokenizers):
    path = Path(path)
    with NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save({'version': 1, 'config': asdict(model.config), 'weights': model.state_dict(),
                    'tokenizers': tokenizers.fingerprints}, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def course_config(tokenizers, positions=5000):
    return ModelConfig(tokenizers.source.vocab_size(), tokenizers.target.vocab_size(), width=512,
                       heads=8, layers=6, ff_width=2048, positions=positions, course_norm=True)


def course_key(key):
    for old, new in (('src_embed', 'source'), ('tgt_embed', 'target')):
        if key == f'{old}.0.lut.weight':
            return f'{new}.tokens.weight'
        if key == f'{old}.1.pe':
            return f'{new}.position'
    if key.startswith('generator.proj.'):
        return key.replace('generator.proj.', 'projection.', 1)
    key = re.sub(r'^(encoder|decoder)\.norm\.', r'\1_norm.', key)
    key = re.sub(r'^(encoder|decoder)\.layers\.', r'\1.', key)
    key = key.replace('.self_attn.', '.self_attention.').replace('.src_attn.', '.cross_attention.')
    key = re.sub(r'\.linears\.([0-3])\.',
                 lambda match: '.' + ('query', 'key', 'value', 'output')[int(match[1])] + '.', key)
    key = key.replace('.feed_forward.w_1.', '.ff.0.').replace('.feed_forward.w_2.', '.ff.3.')
    norms = ('attention_norm', 'ff_norm') if key.startswith('encoder.') else ('attention_norm', 'cross_norm', 'ff_norm')
    key = re.sub(r'\.sublayer\.([0-2])\.norm\.', lambda match: '.' + norms[int(match[1])] + '.', key)
    return re.sub(r'\.(a_2|b_2)$', lambda match: '.weight' if match[1] == 'a_2' else '.bias', key)


def load_model(path, tokenizers, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError('权重文件应为本项目的检查点或课程的 state_dict。')
    if checkpoint.get('version') == 1:
        if checkpoint['tokenizers'] != tokenizers.fingerprints:
            raise ValueError('当前分词模型与检查点的 SHA-256 不一致，请使用训练时对应的两个模型。')
        config = ModelConfig(**checkpoint['config'])
        state = checkpoint['weights']
    elif 'src_embed.1.pe' in checkpoint and 'generator.proj.weight' in checkpoint:
        if (tokenizers.source.vocab_size(), tokenizers.target.vocab_size()) != (32000, 32000):
            raise ValueError('请为课程权重提供课程原配的两个 32000 词分词模型。')
        config = course_config(tokenizers, positions=checkpoint['src_embed.1.pe'].size(1))
        state = {course_key(key): value for key, value in checkpoint.items()}
    else:
        raise ValueError('无法识别检查点格式。')
    model = Transformer(config)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
