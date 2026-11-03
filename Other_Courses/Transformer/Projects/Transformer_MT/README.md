# Transformer 英译中练习

这是跟着《Transformer 算法原理与实战》做的一个小项目。我主要想把课上分开讲的几块串起来，看看一句英文怎么变成 token，怎么送进编码器和解码器，最后怎么生成中文。模型里把 Q、K、V 和注意力掩码单独写出来了，方便对着课件看。

## 先跑个小例子

仓库里有 12 对短句，光用 CPU 就能从分词跑到翻译。下面是在 Windows PowerShell 里从头安装的命令，Python 用 3.12 或 3.13。在项目根目录执行。

```powershell
python -m venv .venv
$py = '.\.venv\Scripts\python.exe'
& $py -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
& $py -m pip install -r requirements.txt
& $py -m transformer_mt demo --device cpu
```

已有能跑的 `.venv` 就直接用它，已有 GPU 版 PyTorch 时只需补齐 `requirements.txt` 里的依赖。下面的命令也用同一个 `$py`。Linux 上对应的解释器是 `.venv/bin/python`。需要配置 CUDA 时可以按自己的显卡选安装命令。[PyTorch 官方安装页](https://pytorch.org/get-started/locally/)

示例会把分词器和权重写到 `runs/demo`，最后可以看到 `I like tea. → 我喜欢喝茶。`。这 12 对句子也拿来做了评估，所以这里的 BLEU 只用来确认流程能跑通。再跑一次时加 `--output runs/demo-2`。

## 用课上的资料试一下

把课程 ZIP 解压到项目根目录的 `Transformer-pytorch` 文件夹。下面先加载课上给的权重翻译一句，再用原语料做一次小规模训练。两条命令用的是同一套分词器。

```powershell
$src = 'Transformer-pytorch/tokenizer/eng.model'
$tgt = 'Transformer-pytorch/tokenizer/chn.model'
$ckpt = 'Transformer-pytorch/run/train/exp/weights/best_bleu_26.30.pth'
& $py -m transformer_mt translate --checkpoint $ckpt --src-tokenizer $src --tgt-tokenizer $tgt --text "The government has implemented various policies to improve the living standards of its citizens."
& $py -m transformer_mt train --preset course --device cuda --train Transformer-pytorch/data/json/train.json --dev Transformer-pytorch/data/json/dev.json --src-tokenizer $src --tgt-tokenizer $tgt --output runs/course-smoke --epochs 1 --batch-size 8 --limit-train 32 --limit-dev 8
```

本机的 RTX 5070 Ti 用 32 对训练句、8 对验证句跑过一轮，也试过用课程权重翻译和在 8 对测试句上算 BLEU。想用全量语料时，去掉 `--limit-train` 和 `--limit-dev`。课程目录在 `.gitignore` 里，下载的课件、数据和权重留在本机。

## 换自己的数据

把训练、验证、测试集分别存成 UTF-8 JSON。每个文件是一组英中句对，比如 `[["I like tea.", "我喜欢喝茶。"]]`。先用训练集做分词器，再训练模型。句子超过 `--max-length` 时会截断并保留结束标记。

```powershell
& $py -m transformer_mt fit-tokenizers --data data/train.json --output runs/tokenizers --vocab-size 8000
$src = 'runs/tokenizers/src.model'
$tgt = 'runs/tokenizers/tgt.model'
& $py -m transformer_mt train --train data/train.json --dev data/dev.json --src-tokenizer $src --tgt-tokenizer $tgt --output runs/first-train --epochs 3 --batch-size 8
$ckpt = 'runs/first-train/best.pt'
& $py -m transformer_mt evaluate --data data/test.json --checkpoint $ckpt --src-tokenizer $src --tgt-tokenizer $tgt
& $py -m transformer_mt translate --text "I like tea." --checkpoint $ckpt --src-tokenizer $src --tgt-tokenizer $tgt
```

每轮的训练和验证损失在 `metrics.json` 里。`best.pt` 取验证损失最低的那一轮，`last.pt` 是最后一轮。拿权重做翻译或评估时，要用和训练时配套的分词模型。

## 看代码和跑测试

想看注意力和掩码，可以从 `transformer_mt/model.py` 开始。`data.py` 负责分词和整理 batch，`engine.py` 里是训练、翻译和评估，`checkpoint.py` 负责保存权重以及兼容课程权重。

```powershell
& $py -m unittest discover -s tests -v
```

本地在 Python 3.12 的 CPU 环境和 Python 3.13 环境各跑过一遍，6 项都通过了。仓库也配了 Windows 和 Linux 的 GitHub Actions 测试。

## 参考

仓库里的实现和示例句是按实战流程另外写的，下面是学习时用到的资料。

- 炮哥带你学的 [《Transformer 算法原理与实战》](https://www.bilibili.com/video/BV1ej1EBWEWu/)。课程资料可以从作者提供的渠道获取。
- Vaswani 等人的 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)。
