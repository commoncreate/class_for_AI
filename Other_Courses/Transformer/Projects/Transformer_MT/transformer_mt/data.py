import hashlib
import io
import json
from pathlib import Path
import sys

import sentencepiece as spm
import torch
from torch.nn.utils.rnn import pad_sequence

PAD, UNK, BOS, EOS = 0, 1, 2, 3


def read_pairs(path, limit=None):
    with Path(path).open(encoding='utf-8-sig') as stream:
        rows = json.load(stream)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f'{path} 应为非空的 [[英文, 中文], ...] JSON 数组。')
    pairs, empty = [], 0
    for index, pair in enumerate(rows[:limit], 1):
        if not isinstance(pair, list) or len(pair) != 2 or not all(isinstance(x, str) for x in pair):
            raise ValueError(f'{path} 的第 {index} 行需要一对英文和中文字符串。')
        if not all(text.strip() for text in pair):
            empty += 1
            continue
        pairs.append(pair)
    if empty:
        print(f'{path} 跳过 {empty} 对空白句子。', file=sys.stderr)
    if not pairs:
        raise ValueError(f'{path} 没有可用句对。')
    return pairs


class Tokenizers:
    def __init__(self, source, target):
        source_bytes, target_bytes = Path(source).read_bytes(), Path(target).read_bytes()
        self.fingerprints = [hashlib.sha256(content).hexdigest() for content in (source_bytes, target_bytes)]
        self.source = spm.SentencePieceProcessor(model_proto=source_bytes)
        self.target = spm.SentencePieceProcessor(model_proto=target_bytes)
        for model in (self.source, self.target):
            if (model.pad_id(), model.unk_id(), model.bos_id(), model.eos_id()) != (PAD, UNK, BOS, EOS):
                raise ValueError('分词器必须使用 PAD=0、UNK=1、BOS=2、EOS=3。')

    @staticmethod
    def encode(model, text, max_length):
        return torch.tensor([BOS, *model.encode(text, out_type=int)[:max_length - 2], EOS], dtype=torch.long)


def fit_tokenizers(pairs, output, vocab_size):
    output = Path(output)
    paths = [output / 'src.model', output / 'tgt.model']
    if any(path.exists() for path in paths):
        raise FileExistsError(f'{output} 已有分词模型，请选择新的输出目录。')
    models = []
    for column in range(2):
        buffer = io.BytesIO()
        spm.SentencePieceTrainer.train(sentence_iterator=(pair[column] for pair in pairs),
            model_writer=buffer, model_type='bpe', vocab_size=vocab_size, hard_vocab_limit=False,
            character_coverage=1.0, pad_id=PAD, unk_id=UNK, bos_id=BOS, eos_id=EOS,
            normalization_rule_name='identity', num_threads=1, shuffle_input_sentence=False, minloglevel=2)
        models.append(buffer.getvalue())
    output.mkdir(parents=True, exist_ok=True)
    for path, content in zip(paths, models):
        with path.open('xb') as stream:
            stream.write(content)
    return paths


def translation_batches(pairs, tokenizers, max_length, batch_size, *, shuffle=False):
    def collate(batch):
        source = [tokenizers.encode(tokenizers.source, pair[0], max_length) for pair in batch]
        target = [tokenizers.encode(tokenizers.target, pair[1], max_length) for pair in batch]
        return pad_sequence(source, batch_first=True, padding_value=PAD), pad_sequence(target, batch_first=True, padding_value=PAD)

    return torch.utils.data.DataLoader(pairs, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
