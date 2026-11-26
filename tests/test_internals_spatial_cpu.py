"""CPU checks for the register-style internal diagnostics of JiT-Spatial-Mamba.

Run from the repository root: PYTHONPATH=. python tests/test_internals_spatial_cpu.py

The CUDA kernel is stubbed with a reference selective scan, so no GPU or
mamba_ssm is needed. On the GPU, scripts/measure_internals_spatial.py compares
the decomposition with the real mixer on every batch and the class-routing
forward with the model's forward.

  A  mixer_quantities rebuilds the mixer (gate on / off, with / without tokens)
  B  spatial_contributions rebuilds the mixer and matches a naive per-source
     transcription (one-hot source, sequential scan, the StateFusion module)
  C  causality: the direct path and the tokens read only earlier scan steps;
     a grid cell reads only what its SASF neighbours have scanned
  D  forward_variant: defaults are the model's forward bit for bit; dropped
     tokens equal a model built without them; token / adaLN labels routed
     separately equal a model whose slots absorb the label difference
  E  branch_norms equal a hand-written block; mixer_dynamics agrees with
     mixer_quantities; pooled tokens pick the right cells
  F  the whole driver runs on a synthetic ImageFolder and writes its outputs
"""

import ast
import json
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
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

from src.internals import block_features, mixer_inputs
from src.internals_spatial import (_scan_inputs, branch_norms, forward_variant, mixer_dynamics, mixer_quantities,
                                   pooled_image_tokens, spatial_contributions)
from src.models.spatial_mamba import JiTSpatialMamba
from src.primitives import modulate

POSITIONS = [[0, 2], [1, 3], [3, 0]]


def tiny_model(tokens=0, gate=False, grid=4, hidden=16, depth=6, start=2, seed=3):
    torch.manual_seed(seed)
    net = JiTSpatialMamba(input_size=grid * 2, patch_size=2, hidden_size=hidden, depth=depth, num_classes=5,
                          bottleneck_dim=8, d_state=1, gate=gate, sasf_dilations=(1, 2), lpu="zero",
                          in_context_len=tokens, in_context_start=start if tokens else 0,
                          in_context_positions=POSITIONS[:tokens] if tokens else None).eval()
    with torch.no_grad():   # zero-init parts (adaLN, LPU, head) and ones-init SASF get real values
        for name, p in net.named_parameters():
            if "A_logs" in name:
                p.add_(torch.randn_like(p) * .3)
            elif "pos_embed" not in name:
                p.add_(torch.randn_like(p) * .15)
    return net


def as_double(net):
    """float64 copy for mixer-level checks. The mixer casts A_logs and the dt bias
    to float32 even in a float64 model, so they are made float32-exact first."""
    net = net.double()
    with torch.no_grad():
        for block in net.blocks:
            for p in (block.mixer.A_logs, block.mixer.dt_projs_bias):
                p.copy_(p.float().double())
    return net


def tiny_inputs(net, batch=3, seed=1):
    torch.manual_seed(seed)
    size = net.input_size
    return torch.randn(batch, 3, size, size), torch.rand(batch), torch.arange(batch) % net.num_classes


def mixer_input(net, block, double=False, seed=2):
    """A random input of the right length for block `block`'s mixer."""
    torch.manual_seed(seed)
    tokens = net.in_context_len if net.in_context_len and block >= net.in_context_start else 0
    x = torch.randn(2, tokens + net.grid_size ** 2, net.hidden_size)
    return x.double() if double else x


def scan_positions(mixer, P, HW):
    order = mixer._scan_order(HW, "cpu")[0] if P else torch.arange(HW)
    return torch.argsort(order)                      # layout index -> scan step


class Quantities(unittest.TestCase):
    def test_rebuilds_mixer(self):
        for gate in (False, True):
            for tokens in (0, 3):
                net = as_double(tiny_model(tokens, gate))
                for block in (0, 4):
                    mixer, x = net.blocks[block].mixer, mixer_input(net, block, double=True)
                    q = mixer_quantities(mixer, x, net.grid_size, net.grid_size)
                    expected = mixer(x, net.grid_size, net.grid_size)
                    self.assertLess(float((q["out"] - expected).abs().max()), 1e-10, (gate, tokens, block))


class Contributions(unittest.TestCase):
    def test_rebuilds_mixer(self):
        for gate in (False, True):
            net = as_double(tiny_model(3, gate))
            for block in (1, 4):
                mixer, x = net.blocks[block].mixer, mixer_input(net, block, double=True)
                _, _, rebuilt = spatial_contributions(mixer, x, 4, 4)
                self.assertLess(float((rebuilt - mixer(x, 4, 4)).abs().max()), 1e-10)

    @torch.no_grad()
    def naive(self, mixer, x, H, W):
        """terms[b, i, j] (d,) by running the scan once per source j, with SASF from the module."""
        q = _scan_inputs(mixer, x, H, W, torch.float64)
        u, delta, A, P = q["u"], q["delta"], q["A"], q["P"]
        batch, length, d = u.shape
        order = scan_positions(mixer, P, H * W)
        steps = torch.argsort(order)                                     # scan step -> layout index
        through = torch.zeros(batch, length, length, d, dtype=torch.float64)
        direct = torch.zeros_like(through)
        for j in range(length):
            states = torch.zeros(batch, length, d, dtype=torch.float64)
            state = torch.zeros(batch, d, dtype=torch.float64)
            for i in steps.tolist():
                state = torch.exp(delta[:, i] * A) * state
                if i == j:
                    state = state + delta[:, i] * q["B"][:, i, None] * u[:, i]
                states[:, i] = state
            grid = states[:, P:].transpose(1, 2).reshape(batch, d, H, W)
            fused = mixer.state_fusion(grid).reshape(batch, d, H * W).transpose(1, 2)
            C = q["C"][..., None]
            through[:, :, j] = torch.cat([states[:, :P], fused], dim=1) * C
            direct[:, :, j] = states * C
        return through, direct

    def test_matches_naive_transcription(self):
        for tokens, block in ((0, 3), (3, 3), (3, 1)):
            net = as_double(tiny_model(tokens))
            mixer, x = net.blocks[block].mixer, mixer_input(net, block, double=True)
            influence, direct, _ = spatial_contributions(mixer, x, 4, 4)
            through, scan_only = self.naive(mixer, x, 4, 4)
            for actual, terms in ((influence, through), (direct, scan_only)):
                expected = terms.norm(dim=-1)
                n = expected.shape[1]
                expected[:, torch.arange(n), torch.arange(n)] = 0
                self.assertLess(float((actual - expected).abs().max()), 1e-10, (tokens, block))
            # the terms (with the diagonal) and D u add up to the pre-norm output
            q = mixer_quantities(mixer, x, 4, 4)
            total = through.sum(dim=2) + q["u"] * mixer.Ds.detach().double()
            self.assertLess(float((total - q["y"]).abs().max()), 1e-10)

    def test_causality(self):
        net = as_double(tiny_model(3))
        block, H = 4, 4
        mixer, x = net.blocks[block].mixer, mixer_input(net, block, double=True)
        influence, direct, _ = spatial_contributions(mixer, x, H, H)
        P = 3
        step = scan_positions(mixer, P, H * H)
        later = step[None, :] > step[:, None]                            # [i, j]: j scanned after i
        self.assertEqual(float(direct[:, later].abs().max()), 0.0)
        self.assertEqual(float(influence[:, :P][:, later[:P]].abs().max()), 0.0)   # tokens are not fused
        # a grid cell i reads j only if some SASF neighbour of i is scanned at or after j
        reach = torch.zeros(P + H * H, P + H * H, dtype=torch.bool)
        for g in range(H * H):
            r, c = divmod(g, H)
            neighbours = {(min(max(r + a * k, 0), H - 1), min(max(c + b * k, 0), H - 1))
                          for k in (1, 2) for a in (-1, 0, 1) for b in (-1, 0, 1)}
            last = max(int(step[P + rr * H + cc]) for rr, cc in neighbours)
            reach[P + g] = step <= last
        reach[:P] = step[None, :] <= step[:P, None]
        self.assertEqual(float(influence[:, ~reach].abs().max()), 0.0)
        self.assertGreater(float(influence[:, P:][:, later[P:]].max()), 0.0)  # SASF does read ahead


class Variants(unittest.TestCase):
    def test_defaults_are_the_forward(self):
        for tokens in (0, 3):
            net = tiny_model(tokens)
            x, t, y = tiny_inputs(net)
            with torch.no_grad():
                self.assertTrue(torch.equal(forward_variant(net, x, t, y), net(x, t, y)))

    def test_dropped_tokens_equal_a_model_without_them(self):
        net = tiny_model(3)
        x, t, y = tiny_inputs(net)
        state = net.state_dict()
        for keep in ([], [0, 2], [1]):
            other = JiTSpatialMamba(input_size=8, patch_size=2, hidden_size=16, depth=6, num_classes=5,
                                    bottleneck_dim=8, d_state=1, gate=False, sasf_dilations=(1, 2), lpu="zero",
                                    in_context_len=len(keep), in_context_start=2 if keep else 0,
                                    in_context_positions=[POSITIONS[e] for e in keep] or None).eval()
            subset = {k: v for k, v in state.items() if k != "incontext_pos_embed"}
            if keep:
                subset["incontext_pos_embed"] = state["incontext_pos_embed"][:, keep]
            other.load_state_dict(subset, strict=True)
            with torch.no_grad():
                expected = other(x, t, y)
            self.assertLess(float((forward_variant(net, x, t, y, keep=keep) - expected).abs().max()), 1e-6, keep)
        # the mixers' token positions are restored afterwards
        self.assertTrue(torch.equal(forward_variant(net, x, t, y), net(x, t, y)))

    def test_separate_routes_equal_shifted_slots(self):
        net = tiny_model(3)
        x, t, _ = tiny_inputs(net)
        for adaln, tokens in ((1, 5), (5, 2), (0, 3)):        # 5 is the null class
            a, b = torch.full((3,), adaln), torch.full((3,), tokens)
            shifted = tiny_model(3)
            with torch.no_grad():
                shifted.incontext_pos_embed += net.y_embedder(b[:1]) - net.y_embedder(a[:1])
                expected = shifted(x, t, a)
            actual = forward_variant(net, x, t, a, b)
            self.assertLess(float((actual - expected).abs().max()), 1e-5, (adaln, tokens))


class Statistics(unittest.TestCase):
    def test_branch_norms_match_hand_written_block(self):
        net = tiny_model(3)
        x, t, y = tiny_inputs(net)
        k, H = 4, 4
        norms = branch_norms(net, x, t, y, [k])
        with torch.no_grad():
            h = block_features(net, x, t, y, [k - 1])[k - 1]
            block = net.blocks[k]
            c = net.t_embedder(t) + net.y_embedder(y)
            shift1, scale1, gate1, shift2, scale2, gate2 = block.adaLN_modulation(c).chunk(6, dim=-1)
            h = h + block._grid_conv(block.cpe1, h, H, H)
            mix = gate1.unsqueeze(1) * block.mixer(modulate(block.norm1(h), shift1, scale1), H, H)
            h = h + mix
            h = h + block._grid_conv(block.cpe2, h, H, H)
            ffn = gate2.unsqueeze(1) * block.mlp(modulate(block.norm2(h), shift2, scale2))
        self.assertLess(float((norms[k][0] - mix.norm(dim=-1)).abs().max()), 1e-5)
        self.assertLess(float((norms[k][1] - ffn.norm(dim=-1)).abs().max()), 1e-5)

    def test_dynamics_agree_with_quantities(self):
        net = tiny_model(3)        # the timestep embedding is float32-only, so the model stays float32
        x, t, y = tiny_inputs(net)
        stats = mixer_dynamics(net, x, t, y, [1, 4], dtype=torch.float64)
        inputs = mixer_inputs(net, x, t, y, [1, 4])
        for block in (1, 4):
            q = mixer_quantities(net.blocks[block].mixer, inputs[block], 4, 4)
            P = q["P"]
            self.assertEqual(stats[block]["tokens"], P)
            self.assertTrue(torch.allclose(stats[block]["retention"], torch.exp(q["delta"] * q["A"]).mean(-1)))
            grid = q["states"][:, P:]
            change = (q["fused"] - grid).flatten(1).norm(dim=1) / grid.flatten(1).norm(dim=1)
            self.assertTrue(torch.allclose(stats[block]["sasf_change"], change))
            self.assertTrue(torch.allclose(stats[block]["sasf_share"].sum(1), torch.ones(3, dtype=torch.float64)))
            self.assertTrue(bool((stats[block]["retention"] > 0).all() and (stats[block]["retention"] < 1).all()))

    def test_pooled_tokens(self):
        grid = torch.zeros(1, 40, 2)
        grid[0, :, 0] = torch.arange(40.)                    # norm grows with the index
        pooled = pooled_image_tokens(grid, .05, torch.Generator().manual_seed(0))
        self.assertEqual(pooled.shape, (1, 4, 2))
        self.assertAlmostEqual(float(pooled[0, 0, 0]), 19.5)
        self.assertAlmostEqual(float(pooled[0, 1, 0]), 38.5)  # top 2 of 40: 39, 38
        self.assertAlmostEqual(float(pooled[0, 3, 0]), 0.5)   # bottom 2: 0, 1


def load_build_model():
    """scripts/evaluate.py::build_model without importing torchvision (not installed here)."""
    source = (ROOT / "scripts" / "evaluate.py").read_text(encoding="utf-8")
    function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "build_model")
    namespace = {}
    exec(compile(ast.Module([function], []), "scripts/evaluate.py", "exec"), namespace)
    return namespace["build_model"]


class Driver(unittest.TestCase):
    def test_end_to_end(self):
        try:
            import matplotlib  # noqa: F401
            import yaml
            from PIL import Image
        except ImportError as error:   # pragma: no cover
            self.skipTest(f"needs matplotlib, pyyaml and Pillow: {error}")
        fake = types.ModuleType("scripts.evaluate")
        fake.build_model = load_build_model()
        sys.modules["scripts.evaluate"] = fake
        import scripts.measure_internals_spatial as driver

        work = Path(tempfile.mkdtemp())
        try:
            rng = np.random.default_rng(0)
            for split, count in (("train", 4), ("val", 3)):
                for label in ("c0", "c1", "c2"):
                    folder = work / "data" / split / label
                    folder.mkdir(parents=True)
                    for i in range(count):
                        pixels = rng.integers(0, 256, (20, 24, 3), dtype=np.uint8)
                        Image.fromarray(pixels).save(folder / f"{label}_{i}.png")
            base = yaml.safe_load((ROOT / "configs/tiny_imagenet/jit-s2-spatial-nogate-incontext-class3.yaml")
                                  .read_text(encoding="utf-8"))
            base["model"].update(input_size=16, patch_size=4, hidden_size=16, depth=6, num_classes=3,
                                 bottleneck_dim=8, sasf_dilations=[1, 2])
            specs = []
            for name, tokens in (("base", 0), ("tok", 2)):
                cfg = json.loads(json.dumps(base))
                cfg["model"].update(in_context_len=tokens, in_context_start=2 if tokens else 0,
                                    in_context_positions=[[0, 2], [2, 1]] if tokens else None)
                (work / f"{name}.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
                torch.manual_seed(len(specs))
                net = fake.build_model(cfg)
                with torch.no_grad():
                    for key, p in net.named_parameters():
                        if "pos_embed" not in key:
                            p.add_(torch.randn_like(p) * .1)
                torch.save({"model": net.state_dict(), "epoch": 7,
                            "ema1": {"net." + k: v.clone() for k, v in net.named_parameters()}},
                           work / f"{name}.pt")
                specs += ["--model", name, str(work / f"{name}.yaml"), str(work / f"{name}.pt")]
            out = work / "out"
            argv = ["measure_internals_spatial.py", *specs, "--data-dir", str(work / "data"), "--out-dir", str(out),
                    "--device", "cpu", "--workers", "0", "--batch-size", "4", "--attn-batch-size", "2",
                    "--stats-per-class", "2", "--probe-train-per-class", "4", "--probe-val-per-class", "3",
                    "--probe-iterations", "10", "--norm-blocks", "2", "4", "--patch-block", "4",
                    "--pca-blocks", "2", "4", "--probe-block", "3", "--attn-blocks", "2", "5",
                    "--attn-images", "4", "--figure-images", "2", "--dynamics-per-class", "1",
                    "--dynamics-t", ".2", ".8", "--ablation-per-class", "1", "--ablation-t", ".3", ".7",
                    "--probe-t", ".3"]
            saved = sys.argv
            sys.argv = argv
            try:
                driver.main()
            finally:
                sys.argv = saved
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            tok = summary["models"]["tok"]
            self.assertLess(tok["influence_rebuild_relative_error"], 1e-4)
            self.assertEqual(len(tok["probes"]["t0.3_null"]["class_tokens"]), 2)
            self.assertIn("tokens_other", tok["ablation"]["variants"])
            self.assertEqual(tok["ablation"]["full_vs_forward_max_abs"], 0.0)
            self.assertNotIn("ablation", summary["models"]["base"])
            for figure in ("fig1_influence_maps.png", "fig2_token_norms.png", "fig4a_tv_ratio.png",
                           "fig5_probes_t0.3.png", "fig6_token_maps_tok.png", "fig7_incontext_tokens.png",
                           "fig8a_retention.png", "fig8b_retention_maps_tok.png", "fig9_sasf.png",
                           "fig10_ablation_tok.png", "fig11_reach_tok.png", "fig12_branch_norms.png",
                           "fig13_outlier_locations.png"):
                self.assertTrue((out / figure).is_file(), figure)
            metrics = np.load(out / "metrics_tok.npz")
            self.assertEqual(metrics["influence_write"].shape, (2, 4, 2, 4, 4))
        finally:
            sys.modules.pop("scripts.evaluate", None)
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
