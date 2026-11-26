"""CPU checks for the register-style internal diagnostics of JiT-VMamba.

Run from the repository root: PYTHONPATH=. python tests/test_internals_cpu.py

The CUDA kernel is stubbed with a reference selective scan, so no GPU or
mamba_ssm is needed. The real kernel is re-checked on every attention batch of
scripts/measure_internals.py, which compares the decomposition with the mixer.
"""

import shutil
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F


def reference_scan(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                   delta_softplus=False, return_last_state=False):
    dtype = u.dtype
    delta = delta.to(dtype)
    if delta_bias is not None:
        delta = delta + delta_bias[None, :, None].to(dtype)
    if delta_softplus:
        delta = F.softplus(delta)
    B = B.repeat_interleave(u.shape[1] // B.shape[1], dim=1).to(dtype)
    C = C.repeat_interleave(u.shape[1] // C.shape[1], dim=1).to(dtype)
    state = torch.zeros(u.shape[0], u.shape[1], A.shape[1], dtype=dtype)
    outputs = []
    for i in range(u.shape[-1]):
        dt = delta[:, :, i, None]
        state = torch.exp(dt * A.to(dtype)) * state + dt * B[:, :, :, i] * u[:, :, i, None]
        outputs.append((state * C[:, :, :, i]).sum(-1))
    y = torch.stack(outputs, -1)
    return y + D[None, :, None].to(dtype) * u if D is not None else y


stub = types.ModuleType("mamba_ssm")
ops = types.ModuleType("mamba_ssm.ops")
interface = types.ModuleType("mamba_ssm.ops.selective_scan_interface")
interface.selective_scan_fn = reference_scan
sys.modules.update({"mamba_ssm": stub, "mamba_ssm.ops": ops,
                    "mamba_ssm.ops.selective_scan_interface": interface})

from src.internals import (block_features, canonical_scan_orders, discover_checkpoint, feature_statistics, fit_probes,
                           influence_maps, mixer_inputs, noised, pca_rgb, scan_contributions,
                           tokens_at, total_variation)
from src.models.vmamba import JiTVMamba, SS2D, cross_scan


def tiny_model(tokens=0, layout="prefix", hidden=24, grid=4, depth=12):
    torch.manual_seed(5)
    net = JiTVMamba(input_size=grid * 2, patch_size=2, hidden_size=hidden, depth=depth, num_classes=7,
                    bottleneck_dim=8, d_state=4, in_context_len=tokens, in_context_start=4 if tokens else 0,
                    in_context_content="class", in_context_layout=layout).eval()
    with torch.no_grad():
        for block in net.blocks:   # adaLN-Zero is zero at init: give the blocks a real gate
            block.adaLN_modulation[-1].bias.normal_(0, .3)
            block.mixer.A_logs.data.clamp_(max=1.0)
    return net


def tiny_inputs(net, batch=3, seed=1):
    torch.manual_seed(seed)
    size = net.input_size
    return torch.randn(batch, 3, size, size), torch.full((batch,), .5), torch.arange(batch)


class ScanOrder(unittest.TestCase):
    def test_baseline_orders_equal_cross_scan(self):
        cells = torch.arange(36).reshape(1, 1, 6, 6).float()
        expected = cross_scan(cells)[0, :, 0].long()
        self.assertTrue(torch.equal(canonical_scan_orders(6, 6, 0, "prefix"), expected))

    def test_prefix_and_row_placement(self):
        grid = cross_scan(torch.arange(16).reshape(1, 1, 4, 4).float())[0, :, 0].long()
        prefix = canonical_scan_orders(4, 4, 2, "prefix")
        for path in range(4):
            self.assertEqual(prefix[path, :2].tolist(), [0, 1])
            self.assertTrue(torch.equal(prefix[path, 2:], grid[path] + 2))
        row = canonical_scan_orders(4, 4, 4, "row")
        for path in range(4):
            self.assertEqual(row[path, ::5].tolist(), [0, 1, 2, 3])
            kept = [i for i in range(20) if i % 5]
            self.assertTrue(torch.equal(row[path, kept], grid[path] + 4))
        for order in (prefix, row):
            self.assertTrue(all(sorted(p.tolist()) == list(range(order.shape[1])) for p in order))

    def test_rejects_unsupported_layout(self):
        with self.assertRaises(ValueError):
            canonical_scan_orders(4, 4, 4, "diagonal")


class Contributions(unittest.TestCase):
    def assert_rebuilds(self, tokens, layout):
        grid = 4
        torch.manual_seed(2)
        mixer = SS2D(d_model=16, d_state=3, in_context_layout=layout).eval()
        x = torch.randn(2, tokens + grid * grid, 16)
        _, output = scan_contributions(mixer, x, grid, grid)
        with torch.no_grad():
            actual = mixer(x, grid, grid)
        self.assertLess(float((output - actual).abs().max() / actual.abs().max()), 1e-5)

    def test_rebuilds_mixer_output(self):
        self.assert_rebuilds(0, "prefix")
        self.assert_rebuilds(1, "prefix")
        self.assert_rebuilds(4, "row")

    def test_matches_naive_transcription(self):
        grid, tokens, d_state = 2, 2, 2
        torch.manual_seed(3)
        mixer = SS2D(d_model=4, d_state=d_state, in_context_layout="row").eval()
        x = torch.randn(1, tokens + grid * grid, 4)
        influence, _ = scan_contributions(mixer, x, grid, grid)
        length, d = x.shape[1], mixer.d_inner
        order = canonical_scan_orders(grid, grid, tokens, "row")
        with torch.no_grad():
            projected = mixer.in_proj(x)
            cells = projected[:, tokens:].view(1, grid, grid, d).permute(0, 3, 1, 2)
            cells = mixer.act(mixer.conv2d(cells)).reshape(1, d, grid * grid).transpose(1, 2)
            joint = torch.cat([projected[:, :tokens], cells], dim=1)[0]
            total = torch.zeros(length, length, d)
            for k in range(4):
                u = joint[order[k]]
                params = u @ mixer.x_proj_weight[k].T
                dt_r, B, C = params[:, :mixer.dt_rank], params[:, mixer.dt_rank:mixer.dt_rank + d_state], \
                    params[:, mixer.dt_rank + d_state:]
                delta = F.softplus(dt_r @ mixer.dt_projs_weight[k].T + mixer.dt_projs_bias[k])
                A = -torch.exp(mixer.A_logs[k])
                for i in range(length):
                    for j in range(i + 1):
                        for c in range(d):
                            coefficient = 0.
                            for s in range(d_state):
                                decay = sum(float(delta[q, c] * A[c, s]) for q in range(j + 1, i + 1))
                                coefficient += float(C[i, s]) * np.exp(decay) * float(delta[j, c] * B[j, s])
                            total[order[k, i], order[k, j], c] += coefficient * u[j, c]
        expected = total.norm(dim=-1)
        expected[range(length), range(length)] = 0
        self.assertLess(float((expected - influence[0]).abs().max() / expected.max()), 1e-4)

    def counts(self, tokens, layout):
        grid = 8
        torch.manual_seed(4)
        mixer = SS2D(d_model=16, d_state=2, in_context_layout=layout).eval()
        influence, _ = scan_contributions(mixer, torch.randn(1, tokens + grid * grid, 16), grid, grid)
        _, read, write = influence_maps(influence, tokens, grid, grid)
        return (read[0] != 0).flatten(1).sum(1).tolist(), (write[0] != 0).flatten(1).sum(1).tolist()

    def test_read_write_counts_match_repository_table(self):
        self.assertEqual(self.counts(8, "row"), ([0, 28, 48, 60, 64, 64, 64, 64], [64, 64, 64, 64, 64, 60, 48, 28]))
        self.assertEqual(self.counts(1, "prefix"), ([0], [64]))

    def test_cell_map_is_mean_over_other_cells(self):
        influence = torch.zeros(1, 6, 6)
        influence[0, 2:, 2 + 3] = 3.0
        influence[0, 2 + 3, 2 + 3] = 0.0
        influence[0, 0, 3] = 7.0     # token reads cell (0, 1)
        influence[0, 4, 1] = 5.0     # token 1 writes to cell (1, 0)
        cells, read, write = influence_maps(influence, 2, 2, 2)
        self.assertAlmostEqual(float(cells[0, 1, 1]), 3.0)
        self.assertEqual(float(read[0, 0, 0, 1]), 7.0)
        self.assertEqual(float(write[0, 1, 1, 0]), 5.0)


class Hooks(unittest.TestCase):
    def test_block_features_match_full_forward_and_stop_early(self):
        net = tiny_model(tokens=1)
        x, t, y = tiny_inputs(net)
        seen, handles = {}, []
        for i in range(12):
            handles.append(net.blocks[i].register_forward_hook(
                lambda _m, _a, out, i=i: seen.__setitem__(i, out.detach().clone())))
        with torch.no_grad():
            net(x, t, y)
        for handle in handles:
            handle.remove()
        calls = []
        last = net.blocks[11].register_forward_pre_hook(lambda *_: calls.append(1))
        captured = block_features(net, x, t, y, [3, 5])
        last.remove()
        self.assertFalse(calls, "block 11 ran although block 5 was the last requested")
        self.assertEqual(captured[3].shape[1], 16)
        self.assertEqual(captured[5].shape[1], 17)
        self.assertTrue(torch.equal(captured[5], seen[5]))
        self.assertFalse(net.blocks[5]._forward_hooks)

    def test_mixer_input_rebuilds_block_mixer(self):
        net = tiny_model(tokens=4, layout="row")
        x, t, y = tiny_inputs(net)
        inputs = mixer_inputs(net, x, t, y, [5])
        _, output = scan_contributions(net.blocks[5].mixer, inputs[5], 4, 4)
        with torch.no_grad():
            actual = net.blocks[5].mixer(inputs[5], 4, 4)
        self.assertLess(float((output - actual).abs().max() / actual.abs().max()), 1e-5)

    def test_tokens_at(self):
        net = tiny_model(tokens=1)
        self.assertEqual([tokens_at(net, i) for i in (3, 4, 11)], [0, 1, 1])
        self.assertEqual(tokens_at(tiny_model(), 11), 0)


class Structural(unittest.TestCase):
    def test_prefix_token_is_image_independent(self):
        net = tiny_model(tokens=1)
        a, t, y = tiny_inputs(net, seed=1)
        b, _, _ = tiny_inputs(net, seed=2)
        fa, fb = block_features(net, a, t, y, [10])[10], block_features(net, b, t, y, [10])[10]
        self.assertEqual(float((fa[:, 0] - fb[:, 0]).abs().max()), 0.0)
        self.assertGreater(float((fa[:, 1:] - fb[:, 1:]).abs().max()), 0.0)

    def test_row_token_zero_is_image_independent_and_last_is_not(self):
        net = tiny_model(tokens=4, layout="row")
        a, t, y = tiny_inputs(net, seed=1)
        b, _, _ = tiny_inputs(net, seed=2)
        fa, fb = block_features(net, a, t, y, [10])[10], block_features(net, b, t, y, [10])[10]
        self.assertEqual(float((fa[:, 0] - fb[:, 0]).abs().max()), 0.0)
        self.assertGreater(float((fa[:, 3] - fb[:, 3]).abs().max()), 0.0)


class Statistics(unittest.TestCase):
    def test_total_variation_known_value(self):
        grid = torch.tensor([[[0., 0.], [3., 4.], [0., 0.], [3., 4.]]])
        # map [[0, v], [0, v]] with |v| = 5: two horizontal steps of 5, two vertical of 0
        self.assertAlmostEqual(float(total_variation(grid, 2, 2)[0]), 2.5)

    def test_interpolant(self):
        x, e = torch.ones(1, 3, 2, 2), 2 * torch.ones(1, 3, 2, 2)
        self.assertTrue(torch.equal(noised(x, e, 0.0, 1.0), e))
        self.assertTrue(torch.equal(noised(x, e, 1.0, 1.0), x))

    def test_feature_statistics_token_slots(self):
        net = tiny_model(tokens=1)
        x, _, y = tiny_inputs(net)
        stats = feature_statistics(net, x, y, torch.randn_like(x), [.1, .5], [3, 5], 1.0)
        self.assertEqual(stats["norm"].shape, (2, 2, 3, 17))
        self.assertTrue(np.isnan(stats["norm"][:, 0, :, 0]).all())
        self.assertTrue(np.isfinite(stats["norm"][:, 1]).all())
        self.assertTrue(np.isfinite(stats["tv"]).all() and (stats["grid_norm"] > 0).all())

    def test_pca_range_and_constant_input(self):
        rgb = pca_rgb(torch.randn(16, 8), 4, 4)
        self.assertEqual(rgb.shape, (4, 4, 3))
        self.assertTrue(0 <= float(rgb.min()) and float(rgb.max()) <= 1)
        self.assertEqual(float(pca_rgb(torch.ones(16, 8), 4, 4).abs().max()), 0.0)

    def test_probes(self):
        torch.manual_seed(0)
        classes, dim, n = 5, 8, 900
        y = torch.randint(classes, (n,))
        means = torch.randn(classes, dim) * 4
        separable = means[y] + torch.randn(n, dim) * .2
        constant = torch.full((n, dim), 3.0)
        noise = torch.randn(n, dim)
        x = torch.stack([separable, constant, noise], dim=1)
        accuracy, decay = fit_probes(x[:600], y[:600], x[600:], y[600:], classes, iterations=300, batch_size=128)
        self.assertGreater(accuracy[0], .97)
        chance = max(float((y[600:] == c).float().mean()) for c in range(classes))
        self.assertLessEqual(accuracy[1], chance + 1e-6)
        self.assertLess(accuracy[2], .35)
        self.assertEqual(decay.shape, (3,))


class CheckpointSearch(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.net = tiny_model(tokens=1)
        self.input = self.root / "input"

    def save(self, relative, net=None):
        path = self.input / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(model=(net or self.net).state_dict(), epoch=7), path)
        return path

    def unpack(self, path, target):
        """What Kaggle does to an uploaded .pt: its zip members become a folder."""
        scratch = path.with_name(path.name + ".tmp")
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            prefix = names[0].split("/")[0] + "/"
            for name in names:
                destination = scratch / name[len(prefix):]
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(name))
        path.unlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        scratch.replace(target)

    def assert_loads_same_weights(self, found):
        state = torch.load(found, map_location="cpu", weights_only=False)["model"]
        reference = self.net.state_dict()
        self.assertEqual(set(state), set(reference))
        for key, value in reference.items():
            self.assertTrue(torch.equal(state[key], value), key)

    def test_finds_plain_file_and_prefers_best(self):
        self.save("run/checkpoint-last.pt")
        best = self.save("run/checkpoint-best.pt")
        self.assertEqual(discover_checkpoint(self.net, self.input), str(best))

    def test_rebuilds_folder_named_pt(self):
        path = self.save("datasets/me/prefix/checkpoint-best.pt")
        self.unpack(path, path)
        found = discover_checkpoint(self.net, self.input, cache=self.root / "cache")
        self.assert_loads_same_weights(found)

    def test_rebuilds_checkpoint_unpacked_at_dataset_root(self):
        path = self.save("datasets/me/checkpoint-best.pt")
        self.unpack(path, self.input / "datasets/me/prefix")
        found = discover_checkpoint(self.net, self.input, cache=self.root / "cache")
        self.assert_loads_same_weights(found)

    def test_hint_filters_unpacked_folders(self):
        path = self.save("datasets/row-run/checkpoint-best.pt")
        self.unpack(path, path)
        with self.assertRaises(ValueError):
            discover_checkpoint(self.net, self.input, "absent", cache=self.root / "cache")
        self.assert_loads_same_weights(discover_checkpoint(self.net, self.input, "row", cache=self.root / "cache"))

    def test_reports_why_a_checkpoint_was_rejected(self):
        self.save("other/checkpoint-best.pt", tiny_model(tokens=2))
        (self.input / "broken").mkdir(parents=True)
        (self.input / "broken/checkpoint-best.pt").write_bytes(b"not a checkpoint")
        with self.assertRaises(ValueError) as caught:
            discover_checkpoint(self.net, self.input)
        message = str(caught.exception)
        self.assertIn("incontext_pos_embed has shape (1, 2, 24)", message)
        self.assertIn("unreadable", message)

    def test_two_matching_runs_are_ambiguous(self):
        self.save("a/checkpoint-best.pt")
        self.save("b/checkpoint-best.pt")
        with self.assertRaises(ValueError):
            discover_checkpoint(self.net, self.input)


if __name__ == "__main__":
    unittest.main(verbosity=2)
