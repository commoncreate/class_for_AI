import argparse
import json
import sys
from pathlib import Path
from pickle import UnpicklingError

import torch

from .checkpoint import course_config, load_model
from .data import Tokenizers, fit_tokenizers, read_pairs
from .engine import choose_device, evaluate_model, train_model, translate_sentences
from .model import ModelConfig


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('参数必须是正整数。')
    return number


def add_runtime(parser):
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')


def add_tokenizers(parser):
    parser.add_argument('--src-tokenizer', required=True, type=Path)
    parser.add_argument('--tgt-tokenizer', required=True, type=Path)


def demo(output, device):
    pairs = [
        ['I like tea.', '我喜欢喝茶。'], ['I like coffee.', '我喜欢喝咖啡。'],
        ['I like books.', '我喜欢读书。'], ['I like music.', '我喜欢听音乐。'],
        ['She likes tea.', '她喜欢喝茶。'], ['She likes coffee.', '她喜欢喝咖啡。'],
        ['She likes books.', '她喜欢读书。'], ['She likes music.', '她喜欢听音乐。'],
        ['He likes tea.', '他喜欢喝茶。'], ['He likes coffee.', '他喜欢喝咖啡。'],
        ['He likes books.', '他喜欢读书。'], ['He likes music.', '他喜欢听音乐。'],
    ]
    output.mkdir(parents=True, exist_ok=False)
    (output / 'pairs.json').write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding='utf-8')
    tokenizers = Tokenizers(*fit_tokenizers(pairs, output / 'tokenizers', 96))
    config = ModelConfig(tokenizers.source.vocab_size(), tokenizers.target.vocab_size(),
                         width=48, layers=1, ff_width=96, dropout=0.0, positions=32)
    checkpoint = train_model(config, tokenizers, pairs, pairs, output / 'training', device,
                             epochs=60, batch_size=6, warmup=30, seed=42)
    model = load_model(checkpoint, tokenizers, device)
    translation, = translate_sentences(model, tokenizers, [pairs[0][0]], device, beam_size=3, max_tokens=24)
    print(f'{pairs[0][0]} → {translation}')
    print('演示样本上的 BLEU（训练样本复用，仅验证流程）')
    result = evaluate_model(model, tokenizers, pairs, device, batch_size=6, beam_size=3, max_tokens=24)
    print(json.dumps(result, ensure_ascii=False))
    print(f'检查点与训练记录位于 {output / "training"}')


def parser_for_project():
    parser = argparse.ArgumentParser(description='Transformer 英译中实战')
    commands = parser.add_subparsers(dest='command', required=True)

    toy = commands.add_parser('demo', help='用自带微型样例运行完整流程')
    toy.add_argument('--output', type=Path, default=Path('runs/demo'))
    add_runtime(toy)

    fit = commands.add_parser('fit-tokenizers', help='从平行文本训练中英文分词器')
    fit.add_argument('--data', required=True, type=Path)
    fit.add_argument('--output', required=True, type=Path)
    fit.add_argument('--vocab-size', type=positive, default=8000)

    train = commands.add_parser('train', help='训练并保存 best.pt 与 last.pt')
    add_tokenizers(train)
    add_runtime(train)
    train.add_argument('--train', required=True, type=Path)
    train.add_argument('--dev', required=True, type=Path)
    train.add_argument('--output', required=True, type=Path)
    train.add_argument('--preset', choices=('small', 'course'), default='small')
    train.add_argument('--epochs', type=positive, default=3)
    train.add_argument('--batch-size', type=positive, default=8)
    train.add_argument('--max-length', type=positive, default=128)
    train.add_argument('--warmup', type=positive, default=1000)
    train.add_argument('--seed', type=int, default=42)
    train.add_argument('--limit-train', type=positive, help='只用前 N 对训练样本，适合快速验证')
    train.add_argument('--limit-dev', type=positive, help='只用前 N 对验证样本')

    for command, help_text in [('translate', '用已有权重翻译'), ('evaluate', '对平行文本计算 SacreBLEU')]:
        inference = commands.add_parser(command, help=help_text)
        add_tokenizers(inference)
        add_runtime(inference)
        inference.add_argument('--checkpoint', required=True, type=Path)
        inference.add_argument('--beam-size', type=positive, default=3)
        inference.add_argument('--max-new-tokens', type=positive, help='默认取 60 与模型位置上限的较小值')
        if command == 'translate':
            inference.add_argument('--text', required=True)
        else:
            inference.add_argument('--data', required=True, type=Path)
            inference.add_argument('--batch-size', type=positive, default=8)
            inference.add_argument('--limit', type=positive, help='只评估前 N 对样本')
    return parser


def execute(args):
    if args.command == 'fit-tokenizers':
        for path in fit_tokenizers(read_pairs(args.data), args.output, args.vocab_size):
            print(path)
        return

    device = choose_device(args.device)
    if device.type == 'cpu':
        torch.set_num_threads(min(4, torch.get_num_threads()))
    if args.command == 'demo':
        demo(args.output, device)
        return

    tokenizers = Tokenizers(args.src_tokenizer, args.tgt_tokenizer)
    if args.command == 'train':
        if args.preset == 'course':
            config = course_config(tokenizers, positions=args.max_length)
        else:
            config = ModelConfig(tokenizers.source.vocab_size(), tokenizers.target.vocab_size(), positions=args.max_length)
        path = train_model(config, tokenizers, read_pairs(args.train, args.limit_train),
                           read_pairs(args.dev, args.limit_dev), args.output, device,
                           epochs=args.epochs, batch_size=args.batch_size, warmup=args.warmup, seed=args.seed)
        print(f'最佳检查点 {path}')
        return

    model = load_model(args.checkpoint, tokenizers, device)
    options = {'beam_size': args.beam_size, 'max_tokens': args.max_new_tokens or min(60, model.config.positions)}
    if args.command == 'translate':
        if not args.text.strip():
            raise ValueError('请输入非空英文句子。')
        translation, = translate_sentences(model, tokenizers, [args.text], device, **options)
        print(translation)
    else:
        metrics = evaluate_model(model, tokenizers, read_pairs(args.data, args.limit), device,
                                 batch_size=args.batch_size, **options)
        print(json.dumps(metrics, ensure_ascii=False))


def main():
    args = parser_for_project().parse_args()
    try:
        execute(args)
    except (OSError, ValueError, RuntimeError, UnpicklingError) as error:
        print(f'运行失败：{error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
