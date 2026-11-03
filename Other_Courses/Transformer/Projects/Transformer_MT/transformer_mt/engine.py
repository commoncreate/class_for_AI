import json
import math
from pathlib import Path

import sacrebleu
import torch
from torch.nn import functional as F
from tqdm import tqdm

from .checkpoint import save_checkpoint
from .data import BOS, EOS, PAD, translation_batches
from .model import Transformer


def choose_device(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('当前 PyTorch 无法使用 CUDA。请检查环境，或指定 --device cpu。')
    return torch.device(name)


def epoch_loss(model, batches, device, optimizer=None, warmup=1, step=0):
    model.train(optimizer is not None)
    total_loss = torch.zeros((), device=device)
    token_count = torch.zeros((), dtype=torch.long, device=device)
    with torch.set_grad_enabled(optimizer is not None):
        for source, target in tqdm(batches, desc='训练' if optimizer else '验证', leave=False, disable=len(batches) < 100):
            source, target = source.to(device), target.to(device)
            labels = target[:, 1:]
            logits = model(source, target[:, :-1])
            loss = F.cross_entropy(logits.flatten(0, 1), labels.reshape(-1), ignore_index=PAD, reduction='sum')
            tokens = labels.ne(PAD).sum()
            if optimizer is not None:
                step += 1
                rate = model.config.width ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)
                for group in optimizer.param_groups:
                    group['lr'] = rate
                optimizer.zero_grad(set_to_none=True)
                (loss / tokens).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
                optimizer.step()
            total_loss += loss.detach()
            token_count += tokens
    mean_loss = (total_loss / token_count).item()
    if not math.isfinite(mean_loss):
        raise RuntimeError('损失不是有限值，请检查输入和训练参数。')
    return mean_loss, step


def train_model(config, tokenizers, train_pairs, dev_pairs, output, device, *, epochs, batch_size, warmup, seed):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    model = Transformer(config).to(device)
    train_batches = translation_batches(train_pairs, tokenizers, config.positions, batch_size, shuffle=True)
    dev_batches = translation_batches(dev_pairs, tokenizers, config.positions, batch_size)
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    history, best_loss, step = [], math.inf, 0
    print(f'训练设备 {device}，训练集 {len(train_pairs)} 对，验证集 {len(dev_pairs)} 对', flush=True)
    for epoch in range(1, epochs + 1):
        train_loss, step = epoch_loss(model, train_batches, device, optimizer, warmup, step)
        dev_loss, _ = epoch_loss(model, dev_batches, device)
        history.append({'epoch': epoch, 'steps': step, 'train_loss': train_loss, 'dev_loss': dev_loss})
        if dev_loss < best_loss:
            best_loss = dev_loss
            save_checkpoint(output / 'best.pt', model, tokenizers)
        if epochs <= 10 or epoch in (1, epochs) or epoch % 10 == 0:
            print(f'epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  dev_loss={dev_loss:.4f}', flush=True)
    save_checkpoint(output / 'last.pt', model, tokenizers)
    (output / 'metrics.json').write_text(json.dumps(history, indent=2) + '\n', encoding='utf-8')
    return output / 'best.pt'


@torch.inference_mode()
def beam_decode(model, source, beam_size, max_tokens):
    if not 1 <= beam_size <= model.config.tgt_vocab - 2:
        raise ValueError('beam size 须为正整数，并小于可生成的 token 数量。')
    if not 1 <= max_tokens <= model.config.positions:
        raise ValueError(f'生成长度必须在 1 到 {model.config.positions} 之间。')
    batch = source.size(0)
    device = source.device
    memory, allowed = model.encode(source)
    memory = memory.repeat_interleave(beam_size, dim=0)
    allowed = allowed.repeat_interleave(beam_size, dim=0)
    sequences = torch.full((batch, beam_size, 1), BOS, dtype=torch.long, device=device)
    scores = torch.full((batch, beam_size), -torch.inf, device=device)
    scores[:, 0] = 0
    completed = torch.zeros_like(scores, dtype=torch.bool)
    vocab = model.config.tgt_vocab
    for _ in range(max_tokens):
        hidden = model.decode(sequences.flatten(0, 1), memory, allowed)
        next_scores = model.projection(hidden[:, -1]).log_softmax(-1).view(batch, beam_size, vocab)
        next_scores[..., PAD] = -torch.inf
        next_scores[..., BOS] = -torch.inf
        # 已结束的序列只能延续 EOS，其累计分数保持原样。
        next_scores = next_scores.masked_fill(completed[..., None], -torch.inf)
        next_scores[..., EOS] = next_scores[..., EOS].masked_fill(completed, 0)
        scores, choices = (scores[..., None] + next_scores).flatten(1).topk(beam_size)
        parents, tokens = choices // vocab, choices % vocab
        history = sequences.gather(1, parents[..., None].expand(-1, -1, sequences.size(-1)))
        sequences = torch.cat((history, tokens[..., None]), dim=-1)
        completed = completed.gather(1, parents) | tokens.eq(EOS)
        if completed.all():
            break
    result = []
    for ids in sequences[:, 0, 1:].tolist():
        result.append(ids[:ids.index(EOS)] if EOS in ids else ids)
    return result


def translate_sentences(model, tokenizers, sentences, device, *, beam_size, max_tokens):
    source = [tokenizers.encode(tokenizers.source, text, model.config.positions) for text in sentences]
    source = torch.nn.utils.rnn.pad_sequence(source, batch_first=True, padding_value=PAD).to(device)
    return [tokenizers.target.decode(ids) for ids in beam_decode(model, source, beam_size, max_tokens)]


def evaluate_model(model, tokenizers, pairs, device, *, batch_size, beam_size, max_tokens):
    predictions = []
    for start in range(0, len(pairs), batch_size):
        sentences = [source for source, _ in pairs[start:start + batch_size]]
        predictions.extend(translate_sentences(model, tokenizers, sentences, device,
                                               beam_size=beam_size, max_tokens=max_tokens))
    references = [target for _, target in pairs]
    metric = sacrebleu.metrics.BLEU(tokenize='zh')
    score = metric.corpus_score(predictions, [references])
    return {'sentences': len(pairs), 'bleu': round(score.score, 4), 'signature': str(metric.get_signature())}
