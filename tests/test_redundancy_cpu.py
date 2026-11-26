"""CPU checks: true bypass, restoration, pairing, CFG, and real JiM prefix wiring.

Run from the repository root: PYTHONPATH=. python tests/test_redundancy_cpu.py
"""

import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from src.redundancy import (bypass_blocks, image_noise, measure_batch, predictions,
                            removal_variants, summarize)
from scripts.measure_redundancy import load_weights, validation_records


class Add(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(value)))
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x + self.value


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Add(1), Add(2), Add(3)])

    def forward(self, x, t=None, y=None):
        for block in self.blocks:
            x = block(x)
        return x if y is None else x + y[:, None, None, None]


def scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None,
             delta_softplus=False, **kwargs):
    delta = delta.float() + delta_bias[None, :, None]
    delta = F.softplus(delta) if delta_softplus else delta
    B = B.repeat_interleave(u.shape[1] // B.shape[1], dim=1).float()
    C = C.repeat_interleave(u.shape[1] // C.shape[1], dim=1).float()
    state = torch.zeros(u.shape[0], u.shape[1], A.shape[1])
    result = []
    for i in range(u.shape[-1]):
        dt = delta[:, :, i, None]
        state = torch.exp(dt * A) * state + dt * B[:, :, :, i] * u[:, :, i, None]
        result.append((state * C[:, :, :, i]).sum(-1))
    out = torch.stack(result, -1)
    return out + D[None, :, None] * u if D is not None else out


class RedundancyTests(unittest.TestCase):
    def test_bypass_and_exception_restoration(self):
        net = Toy().eval()
        original = tuple(net.blocks)
        x = torch.zeros(2, 1, 2, 2)
        expected = net(x)
        with bypass_blocks(net, ()):
            self.assertTrue(torch.equal(net(x), expected))
        calls = original[1].calls
        with bypass_blocks(net, (1,)):
            self.assertTrue(torch.equal(net(x), torch.full_like(x, 4)))
        self.assertEqual(original[1].calls, calls)
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with bypass_blocks(net, (0, 2)):
                raise RuntimeError("deliberate")
        self.assertEqual(tuple(net.blocks), original)
        self.assertTrue(torch.equal(net(x), expected))
        for indices in ((-1,), (3,), (1, 1)):
            with self.assertRaises(ValueError):
                with bypass_blocks(net, indices):
                    pass

    def test_variants_and_noise(self):
        variants = removal_variants(12)
        self.assertEqual(len(variants), 33)  # baseline + 12 singles + 11 pairs + 9 quartets
        self.assertEqual(variants["skip_8_9_10_11"], (8, 9, 10, 11))
        a = image_noise((3, 4, 4), "class/a", 42)
        self.assertTrue(torch.equal(a, image_noise((3, 4, 4), "class/a", 42)))
        self.assertFalse(torch.equal(a, image_noise((3, 4, 4), "class/b", 42)))

    def test_cfg_boundaries_and_paired_metrics(self):
        net = Toy().eval()
        x = torch.zeros(3, 1, 2, 2)
        labels = torch.tensor([0, 1, 2])
        conditional, guided = predictions(net, x, torch.tensor([.1, .5, 1.]), labels, 3, 2.5, [.1, 1.])
        self.assertTrue(torch.equal(conditional[[0, 2]], guided[[0, 2]]))
        self.assertTrue(torch.equal(guided[1], torch.full_like(x[1], 4)))
        variants = removal_variants(3, (1,))
        kwargs = dict(num_classes=3, noise_scale=1., t_eps=.05, cfg_scale=2.5, cfg_interval=[.1, 1.])
        noise = torch.randn_like(x)
        errors, drift = measure_batch(net, x, labels, noise, [.2, .98], variants, **kwargs)
        self.assertTrue(np.array_equal(drift[0], np.zeros_like(drift[0])))
        np.testing.assert_allclose(drift[1, 0], 1, atol=1e-6)
        split = [measure_batch(net, x[i:i+1], labels[i:i+1], noise[i:i+1], [.2, .98], variants, **kwargs)
                 for i in range(3)]
        np.testing.assert_allclose(errors, np.concatenate([a for a, b in split], axis=-1))
        rows, ranking = summarize(errors, drift, variants, [.2, .98], .05)
        r = next(r for r in rows if r['variant'] == 'skip_0' and r['mode'] == 'conditional' and r['t'] == .98)
        self.assertAlmostEqual(r['v_drift_mse'], 400, places=3)
        base = next(r for r in ranking if r['variant'] == 'baseline')
        self.assertEqual(base['mean_delta_v_mse'], 0)
        self.assertEqual(base['paired_image_sem'], 0)
        row = next(r for r in ranking if r['variant'] == 'skip_0' and r['mode'] == 'conditional')
        delta = ((errors[1, 0] - errors[0, 0]) / np.array([.8**2, .05**2])[:, None]).mean(0)
        self.assertAlmostEqual(row['paired_image_sem'], np.std(delta, ddof=1) / np.sqrt(3))

    def test_checkpoint_and_flat_validation(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            net = Toy().eval()
            path = root / "checkpoint.pt"
            torch.save(dict(model=net.state_dict(), ema1={"net."+k: torch.full_like(v, 9)
                        for k, v in net.named_parameters()}), path)
            load_weights(net, path, 1)
            self.assertTrue(all(p.item() == 9 for p in net.parameters()))
            with self.assertRaises(ValueError):
                load_weights(net, path, 2)
            for label in ('a', 'b'):
                (root / 'train' / label).mkdir(parents=True)
            (root / 'val' / 'images').mkdir(parents=True)
            for name in ('a.jpg', 'b.jpg'):
                (root / 'val' / 'images' / name).touch()
            (root / 'val' / 'val_annotations.txt').write_text('a.jpg a\nb.jpg b\n')
            records, classes = validation_records(root, 2, 1, 42)
            self.assertEqual(classes, ['a', 'b'])
            self.assertEqual([r[1] for r in records], [0, 1])

    def test_real_jim_preserves_prefix_insertion_and_weights(self):
        iface = types.ModuleType('mamba_ssm.ops.selective_scan_interface')
        iface.selective_scan_fn = scan_ref
        sys.modules['mamba_ssm'] = types.ModuleType('mamba_ssm')
        sys.modules['mamba_ssm.ops'] = types.ModuleType('mamba_ssm.ops')
        sys.modules[iface.__name__] = iface
        from src.models.vmamba import JiTVMamba
        for layout, length in [('prefix', 0), ('prefix', 1), ('row', 2)]:
            torch.manual_seed(4)
            net = JiTVMamba(input_size=8, patch_size=4, hidden_size=32, depth=4,
                           num_heads=2, mlp_ratio=2., bottleneck_dim=8, num_classes=3,
                           d_state=2, in_context_len=length, in_context_start=1,
                           in_context_content='class', in_context_layout=layout).eval()
            # Nonzero gates/head are essential: zero-init would hide every bypass bug.
            with torch.no_grad():
                net.final_layer.linear.weight.normal_(std=.1)
                for block in net.blocks:
                    block.adaLN_modulation[-1].bias.normal_(std=.15)
            x, t, y = torch.randn(2, 3, 8, 8), torch.tensor([.2, .8]), torch.tensor([0, 1])
            before = {k: v.clone() for k, v in net.state_dict().items()}
            seen = []
            handle = net.blocks[2].register_forward_pre_hook(lambda module, inputs: seen.append(inputs[0].shape[1]))
            with torch.inference_mode():
                reference = net(x, t, y)
                with bypass_blocks(net, ()):
                    self.assertTrue(torch.equal(net(x, t, y), reference))
                with bypass_blocks(net, (1,)):
                    changed = net(x, t, y)
                self.assertEqual(changed.shape, reference.shape)
                self.assertGreater((changed-reference).abs().max().item(), 0)
                self.assertTrue(torch.equal(net(x, t, y), reference))
            handle.remove()
            self.assertTrue(all(count == 4 + length for count in seen))
            self.assertTrue(all(torch.equal(before[k], v) for k, v in net.state_dict().items()))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
