"""
tests/test_resume_cpu.py
========================
End-to-end CPU verification of resuming in scripts/run_experiment.py
(RNG-RESUME): the real main() trains a tiny JiT on a stand-in CIFAR-10
(shuffled DataLoader, 2 worker processes doing a random flip, label dropout),
uninterrupted and as a chain of jobs.

The seed is set once at start-up. Before the fix checkpoints held no RNG
state, so every resumed job replayed the run's first epochs exactly: same
data order, flips, t, noise and label drops.

Checks:
  A. uninterrupted 4 epochs == 2 epochs + resume to 4: the training stream
     (images, labels, t per step) and every tensor of the final checkpoints
     (model, both EMAs, optimizer), losses, best_loss and metrics.json are
     identical.
  B. the previous training script (git HEAD~ before this fix, exec'd) replays:
     its resumed epochs 3-4 are exactly epochs 1-2 again.
  C. a checkpoint without RNG state (written before the fix) resumes on a
     fresh stream derived from (seed, rank, epoch): no replay, and says so.
  D. best_loss is updated before saving: checkpoint-last carries min(losses).
  E. metrics.json is written every epoch and carried across the resume.
  F. a missing resume_from warns loudly and trains from scratch.
  G. save_checkpoint is atomic: a write killed half-way leaves the previous
     checkpoint byte-identical and loadable; normal runs leave no .tmp.
  H. restore_rng_states picks this rank's state, and falls back to distinct
     (seed, rank, epoch) seeds on a missing state / rank-count mismatch.

Run from the repo root:
    python tests/test_resume_cpu.py
(no PYTHONPATH needed -- the repo root is derived from this file's location)
"""
import os
import sys

import torch
from torch.utils.data import Dataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_IMAGES, BATCH = 32, 8                         # 4 steps per epoch


# ── stand-in torchvision pieces (module level: DataLoader workers unpickle them)
class Compose:
    def __init__(self, ts):
        self.ts = ts

    def __call__(self, x):
        for t in self.ts:
            x = t(x)
        return x


class RandomHorizontalFlip:                     # draws from the worker's torch RNG
    def __call__(self, x):
        return x.flip(-1) if torch.rand(1).item() < 0.5 else x


class ToTensor:
    def __call__(self, x):
        return x


class Normalize:
    def __init__(self, mean, std):
        pass

    def __call__(self, x):
        return (x - 0.5) / 0.5


class FakeCIFAR10(Dataset):
    def __init__(self, root=None, train=True, download=False, transform=None):
        g = torch.Generator().manual_seed(0)
        self.data = torch.rand(N_IMAGES, 3, 8, 8, generator=g)
        self.labels = [i % 10 for i in range(N_IMAGES)]
        self.transform = transform

    def __len__(self):
        return N_IMAGES

    def __getitem__(self, i):
        return self.transform(self.data[i]), self.labels[i]


def main():
    import contextlib
    import copy
    import hashlib
    import importlib.util
    import io
    import json
    import shutil
    import subprocess
    import tempfile
    import types
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, REPO_ROOT)

    # stubs for the script's imports (no torchvision / PyYAML / tqdm needed).
    # Each gets a ModuleSpec: torch._dynamo looks third-party modules up.
    from importlib.machinery import ModuleSpec

    def stub(name):
        m = types.ModuleType(name)
        m.__spec__ = ModuleSpec(name, None)
        return m

    tv = stub("torchvision")
    tv.datasets = stub("torchvision.datasets")
    tv.datasets.CIFAR10 = FakeCIFAR10
    tv.transforms = stub("torchvision.transforms")
    for c in (Compose, RandomHorizontalFlip, ToTensor, Normalize):
        setattr(tv.transforms, c.__name__, c)
    sys.modules.update({"torchvision": tv, "torchvision.datasets": tv.datasets,
                        "torchvision.transforms": tv.transforms})
    try:
        import yaml                                                  # noqa: F401
    except ImportError:
        sys.modules["yaml"] = stub("yaml")
    try:
        import tqdm                                                  # noqa: F401
    except ImportError:
        class _Bar:
            def __init__(self, it, **kw):
                self.it = it

            def __iter__(self):
                return iter(self.it)

            def set_postfix(self, **kw):
                pass

            def close(self):
                pass

        tq = stub("tqdm")
        tq.tqdm = _Bar
        sys.modules["tqdm"] = tq

    import src.train as T
    import src.utils as U

    # record the training stream: images, labels and t of every step
    STREAM = []
    orig_forward, orig_sample_t = T.Denoiser.forward, T.Denoiser.sample_t

    def rec_forward(self, x, labels):
        STREAM.append(("x", round(x.double().sum().item(), 6), tuple(labels.tolist())))
        return orig_forward(self, x, labels)

    def rec_sample_t(self, n, device=None):
        t = orig_sample_t(self, n, device=device)
        STREAM.append(("t", tuple(round(v, 6) for v in t.tolist())))
        return t

    T.Denoiser.forward, T.Denoiser.sample_t = rec_forward, rec_sample_t

    def load_script(src, name):
        mod = types.ModuleType(name)
        mod.__file__ = os.path.join(REPO_ROOT, "scripts", "run_experiment.py")
        exec(compile(src, mod.__file__, "exec"), mod.__dict__)
        return mod

    new_script = load_script(open(os.path.join(REPO_ROOT, "scripts", "run_experiment.py"),
                                  encoding="utf-8").read(), "run_new")

    tmp = tempfile.mkdtemp()
    CFG = {
        "experiment": {"name": "resume-test", "model": "jit", "dataset": "cifar10",
                       "data_dir": tmp, "seed": 42},
        "model": {"input_size": 8, "patch_size": 2, "in_channels": 3, "hidden_size": 32,
                  "depth": 2, "num_heads": 2, "mlp_ratio": 4.0, "bottleneck_dim": 8,
                  "num_classes": 10, "attn_drop": 0.0, "proj_drop": 0.0,
                  "in_context_len": 0, "in_context_start": 0},
        "training": {"epochs": 4, "batch_size": BATCH, "warmup_epochs": 1, "blr": 5e-2,
                     "optimizer": "adamw", "betas": [0.9, 0.95], "weight_decay": 0.0,
                     "ema_decay1": 0.9999, "ema_decay2": 0.9996, "amp": False,
                     "num_workers": 2},
        "diffusion": {"P_mean": -0.8, "P_std": 0.8, "t_eps": 0.05, "noise_scale": 1.0},
        "cfg": {"label_drop_prob": 0.1, "cfg_scale": 1.0, "cfg_interval": [0.0, 1.0]},
        "sampling": {"method": "heun", "steps": 50},
        "checkpoint": {"save_last_freq": 1, "save_archive_freq": 2, "resume_from": None,
                       "output_dir": None},
    }

    def run(script, out, epochs, resume=None):
        cfg = copy.deepcopy(CFG)
        cfg["training"]["epochs"] = epochs
        cfg["checkpoint"]["output_dir"] = out
        cfg["checkpoint"]["resume_from"] = resume
        script.load_config = lambda path: cfg
        sys.argv = ["run_experiment.py", "--config", "unused.yaml"]
        start = len(STREAM)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            script.main()
        return STREAM[start:], buf.getvalue()

    per_epoch = 2 * (N_IMAGES // BATCH)                  # 2 records ("x", "t") per step

    def epochs_of(stream):
        return [stream[i:i + per_epoch] for i in range(0, len(stream), per_epoch)]

    def ckpt(path):
        return torch.load(path, map_location="cpu", weights_only=False)

    def same_ckpt(a, b):
        for key in ("model", "ema1", "ema2"):
            assert a[key].keys() == b[key].keys()
            for k in a[key]:
                assert torch.equal(a[key][k], b[key][k]), (key, k)
        sa, sb = a["optimizer"]["state"], b["optimizer"]["state"]
        assert sa.keys() == sb.keys()
        for i in sa:
            for k in sa[i]:
                assert torch.equal(torch.as_tensor(sa[i][k]), torch.as_tensor(sb[i][k])), (i, k)
        assert a["losses"] == b["losses"] and a["best_loss"] == b["best_loss"]
        assert a["epoch"] == b["epoch"] and a["global_step"] == b["global_step"]

    try:
        # ═══ A. uninterrupted == chain ════════════════════════════════════════
        A, B = os.path.join(tmp, "A"), os.path.join(tmp, "B")
        s_full, _ = run(new_script, A, 4)
        s_job1, _ = run(new_script, B, 2)
        metrics_after_job1 = json.load(open(os.path.join(B, "metrics.json")))
        s_job2, log2 = run(new_script, B, 4, resume=os.path.join(B, "checkpoint-last.pt"))
        assert "RNG restored exactly" in log2, log2
        assert len(s_full) == 4 * per_epoch and s_job1 + s_job2 == s_full
        for name in ("checkpoint-last.pt", "checkpoint-best.pt", "checkpoint-ep004.pt"):
            same_ckpt(ckpt(os.path.join(A, name)), ckpt(os.path.join(B, name)))
        m_a = json.load(open(os.path.join(A, "metrics.json")))
        m_b = json.load(open(os.path.join(B, "metrics.json")))
        assert m_a == m_b and [m["step"] for m in m_b] == [1, 2, 3, 4]
        print("A  uninterrupted 4 epochs == 2 + resume to 4: training stream (images, "
              "labels, t), model, EMAs, optimizer, losses, best_loss, metrics all identical")

        # ═══ B. the previous script replays epochs 1-2 after a resume ═════════
        prev_src = subprocess.run(["git", "show", "4589ee1:scripts/run_experiment.py"],
                                  cwd=REPO_ROOT, capture_output=True, check=True
                                  ).stdout.decode("utf-8")
        prev_script = load_script(prev_src.replace("num_workers = 12,", "num_workers = 2,"),
                                  "run_prev")
        C_ = os.path.join(tmp, "C")
        p1, _ = run(prev_script, C_, 2)
        p2, _ = run(prev_script, C_, 4, resume=os.path.join(C_, "checkpoint-last.pt"))
        assert p1 == s_full[:2 * per_epoch], "previous script's first job differs"
        assert p2 == p1, "previous script did not replay"
        assert p2 != s_full[2 * per_epoch:]
        print("B  previous script (4589ee1): resumed epochs 3-4 == epochs 1-2 exactly "
              "(the replay); fixed script: == uninterrupted epochs 3-4")

        # ═══ C. checkpoint without RNG state: fresh stream, no replay ═════════
        old = ckpt(os.path.join(B, "checkpoint-ep002.pt"))
        old.pop("rng")
        old.pop("metrics")
        D = os.path.join(tmp, "D")
        os.makedirs(D)
        torch.save(old, os.path.join(D, "old.pt"))
        s_fb, log_fb = run(new_script, D, 4, resume=os.path.join(D, "old.pt"))
        assert "reseeded from (seed, rank, epoch)" in log_fb, log_fb
        e1, e2 = epochs_of(s_full)[:2]
        f3, f4 = epochs_of(s_fb)
        assert f3 not in (e1, e2) and f4 not in (e1, e2), "fallback replayed the start"
        print("C  checkpoint without RNG state: resumes on a fresh (seed, rank, epoch) "
              "stream, no replay of epochs 1-2, and reports it")

        # ═══ D. best_loss carried by checkpoint-last ══════════════════════════
        for d in (A, B):
            last = ckpt(os.path.join(d, "checkpoint-last.pt"))
            assert last["best_loss"] == min(last["losses"])
            assert ckpt(os.path.join(d, "checkpoint-best.pt"))["best_loss"] == min(last["losses"])
        print("D  checkpoint-last carries best_loss == min(losses), so a resume never "
              "overwrites checkpoint-best with a worse epoch")

        # ═══ E. metrics.json every epoch, carried across the resume ═══════════
        assert [m["step"] for m in metrics_after_job1] == [1, 2], metrics_after_job1
        assert ckpt(os.path.join(B, "checkpoint-ep002.pt"))["metrics"] == metrics_after_job1
        print("E  metrics.json written every epoch (after job 1: epochs 1-2, on disk "
              "although training was not finished) and carried on: final epochs 1-4")

        # ═══ F. missing resume_from ═══════════════════════════════════════════
        E_ = os.path.join(tmp, "E")
        s_miss, log_miss = run(new_script, E_, 1, resume=os.path.join(E_, "nope.pt"))
        assert "WARNING" in log_miss and "TRAINING FROM SCRATCH" in log_miss, log_miss
        assert s_miss == s_full[:per_epoch]
        print("F  missing resume_from: loud WARNING, trains from scratch (epoch 1 stream)")

        # ═══ G. atomic checkpoint writes ══════════════════════════════════════
        leftovers = [f for d in (A, B, D, E_) for f in os.listdir(d) if f.endswith(".tmp")]
        assert not leftovers, leftovers
        last = os.path.join(B, "checkpoint-last.pt")
        before = hashlib.sha256(open(last, "rb").read()).hexdigest()
        real_save = U.torch.save

        def killed_save(obj, path):
            with open(path, "wb") as f:
                f.write(b"half a checkpoint")
            raise RuntimeError("SIGKILL at the 48 h limit")

        net = T.Denoiser(net=torch.nn.Linear(2, 2), img_size=8, num_classes=10)
        net.init_ema()
        opt = torch.optim.AdamW(net.parameters())
        U.torch.save = killed_save
        try:
            U.save_checkpoint(last, 5, 20, net, opt, [1.0], 1.0)
            raise AssertionError("killed save did not raise")
        except RuntimeError:
            pass
        finally:
            U.torch.save = real_save
        assert hashlib.sha256(open(last, "rb").read()).hexdigest() == before
        assert ckpt(last)["epoch"] == 4
        print("G  no .tmp left by normal runs; a write killed half-way leaves "
              "checkpoint-last.pt byte-identical and loadable")

        # ═══ H. restore_rng_states ════════════════════════════════════════════
        torch.manual_seed(1); s0 = U.get_rng_states()
        torch.manual_seed(2); s1 = U.get_rng_states()
        assert U.restore_rng_states([s0, s1], rank=1, world_size=2, seed=42, start_epoch=3)
        assert torch.equal(torch.get_rng_state(), s1["torch"])
        seen = set()
        for states, ws in ((None, 1), ([s0], 2)):
            for rank in range(ws):
                for ep in (2, 3):
                    assert not U.restore_rng_states(states, rank, ws, 42, ep)
                    seen.add(torch.get_rng_state().sum().item())
        torch.manual_seed(42)
        assert torch.get_rng_state().sum().item() not in seen
        print("H  restore_rng_states: rank's own state when available; otherwise "
              "distinct (seed, rank, epoch) seeds, never the start-up seed")
    finally:
        T.Denoiser.forward, T.Denoiser.sample_t = orig_forward, orig_sample_t
        shutil.rmtree(tmp, ignore_errors=True)

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
