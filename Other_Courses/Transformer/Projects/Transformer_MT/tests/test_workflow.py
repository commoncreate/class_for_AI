import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import torch

from transformer_mt.checkpoint import load_model
from transformer_mt.data import Tokenizers, read_pairs
from transformer_mt.model import ModelConfig, Transformer


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.model = Transformer(ModelConfig(24, 32, width=16, heads=2, layers=1, ff_width=32,
                                            dropout=0, positions=12)).eval()

    @torch.inference_mode()
    def test_future_target_tokens_do_not_change_earlier_logits(self):
        source = torch.tensor([[2, 5, 8, 3]])
        first = self.model(source, torch.tensor([[2, 4, 6, 7]]))
        second = self.model(source, torch.tensor([[2, 4, 9, 10]]))
        torch.testing.assert_close(first[:, :2], second[:, :2])

    @torch.inference_mode()
    def test_source_padding_does_not_change_decoder_logits(self):
        target = torch.tensor([[2, 4, 6]])
        first = self.model(torch.tensor([[2, 5, 8, 3]]), target)
        second = self.model(torch.tensor([[2, 5, 8, 3, 0, 0]]), target)
        torch.testing.assert_close(first, second, rtol=1e-5, atol=2e-6)


class DataTests(unittest.TestCase):
    def test_blank_pairs_are_skipped_but_malformed_pairs_fail(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'pairs.json'
            path.write_text(json.dumps([['', '空'], ['Tea.', '茶。']]), encoding='utf-8')
            self.assertEqual(read_pairs(path), [['Tea.', '茶。']])
            path.write_text(json.dumps([['Tea.', 1]]), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '第 1 行'):
                read_pairs(path)


class DemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        cls.output = Path(directory.name) / '演示'
        cls.result = cls.run_cli('demo', '--device', 'cpu', '--output', str(cls.output))

    @staticmethod
    def run_cli(*args):
        return subprocess.run([sys.executable, '-X', 'utf8', '-m', 'transformer_mt', *args],
                              cwd=Path(__file__).resolve().parents[1], capture_output=True,
                              encoding='utf-8', timeout=120, check=False)

    def test_cpu_pipeline_learns_and_translates_without_course_assets(self):
        self.assertEqual(self.result.returncode, 0, self.result.stdout + self.result.stderr)
        self.assertIn('I like tea. → 我喜欢喝茶。', self.result.stdout)
        history = json.loads((self.output / 'training/metrics.json').read_text(encoding='utf-8'))
        self.assertLess(history[-1]['dev_loss'], history[0]['dev_loss'] / 5)
        self.assertTrue((self.output / 'training/last.pt').is_file())

    def test_checkpoint_rejects_swapped_tokenizers(self):
        self.assertEqual(self.result.returncode, 0, self.result.stderr)
        tokenizers = Tokenizers(self.output / 'tokenizers/tgt.model', self.output / 'tokenizers/src.model')
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            load_model(self.output / 'training/best.pt', tokenizers, 'cpu')

    def test_saved_checkpoint_can_be_used_by_translate_command(self):
        self.assertEqual(self.result.returncode, 0, self.result.stderr)
        result = self.run_cli('translate', '--device', 'cpu', '--text', 'I like tea.',
                              '--checkpoint', str(self.output / 'training/best.pt'),
                              '--src-tokenizer', str(self.output / 'tokenizers/src.model'),
                              '--tgt-tokenizer', str(self.output / 'tokenizers/tgt.model'))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), '我喜欢喝茶。')
