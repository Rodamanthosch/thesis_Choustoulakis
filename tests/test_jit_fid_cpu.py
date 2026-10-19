"""
tests/test_jit_fid_cpu.py
=========================
CPU-only verification of the official ImageNet FID path (src/utils.py:
uses_jit_fid, jit_fid_stats_path, ensure_jit_torch_fidelity,
evaluate_fid_is_jit) used by scripts/evaluate.py.

No network, no pip, no GPU: the download and the pip call are intercepted, and
torch-fidelity is a fake package written to a temp dir (stock first, then the
"installed" fork), so the real sys.modules purge + re-import logic runs.

Checks:
  A. protocol selection: every configs/imagenet/*.yaml -> official JiT FID;
     every Tiny-ImageNet / CIFAR-10 config -> folder FID (unchanged); and
     ImageNet subsets / other resolutions stay on the folder path.
  B. stats: first call downloads from the PINNED JiT commit via a .part file;
     sha256 is checked on every call (a corrupted file raises); the pinned
     hashes are 64-hex and the real files are verified if present locally.
  C. fork install: a stock torch-fidelity triggers exactly
     `pip install -q --force-reinstall --no-deps <fork @ 40-hex sha>`, the
     stale module is purged and the fork re-imported; an already-present fork
     triggers no pip call; a failed install raises instead of falling back.
  D. evaluate_fid_is_jit calls calculate_metrics with EXACTLY the kwargs of
     LTH14/JiT engine_jit.py evaluate(), and maps the result keys.
  E. evaluate.py wiring: --n_samples defaults to None (resolved to 50K for
     ImageNet-1k, 10K otherwise) and the JiT path is imported and used.

Run from the repo root:
    python tests/test_jit_fid_cpu.py
(no PYTHONPATH needed -- the repo root is derived from this file's location)
"""
import ast
import glob
import hashlib
import os
import re
import shutil
import sys
import tempfile
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

try:                                    # src.utils imports yaml for load_config only
    import yaml
    HAVE_YAML = True
except ImportError:
    sys.modules["yaml"] = types.ModuleType("yaml")
    HAVE_YAML = False

import src.utils as U                                       # noqa: E402


def read_cfg(path):
    """dataset / input_size / num_classes of a config (regex if no PyYAML)."""
    text = open(path, encoding="utf-8").read()
    if HAVE_YAML:
        return yaml.safe_load(text)
    get = lambda k: re.search(rf"^\s*{k}:\s*([^\s#]+)", text, re.M).group(1)  # noqa: E731
    return {"experiment": {"dataset": get("dataset")},
            "model": {"input_size": int(get("input_size")),
                      "num_classes": int(get("num_classes"))}}


# ═══ A. protocol selection ═══════════════════════════════════════════════════
imagenet = sorted(glob.glob(os.path.join(REPO_ROOT, "configs", "imagenet", "*.yaml")))
others = sorted(glob.glob(os.path.join(REPO_ROOT, "configs", "tiny_imagenet", "*.yaml")) +
                [p for p in glob.glob(os.path.join(REPO_ROOT, "configs", "cifar10", "*.yaml"))
                 if "hparam-sweep" not in p])          # the sweep file has no model block
assert imagenet and others
for p in imagenet:
    assert U.uses_jit_fid(read_cfg(p)), f"{p} should use the JiT protocol"
for p in others:
    assert not U.uses_jit_fid(read_cfg(p)), f"{p} must keep the folder FID"
mk = lambda ds, s, c: {"experiment": {"dataset": ds},                       # noqa: E731
                       "model": {"input_size": s, "num_classes": c}}
assert U.uses_jit_fid(mk("imagenet", 512, 1000))
for cfg in (mk("imagenet", 256, 100), mk("imagenet", 128, 1000), mk("imagenet", 64, 200),
            mk("cifar10", 32, 10)):
    assert not U.uses_jit_fid(cfg), cfg
print(f"A  JiT FID for all {len(imagenet)} ImageNet-256 configs; folder FID kept for "
      f"{len(others)} Tiny-ImageNet / CIFAR-10 configs and for ImageNet subsets")


# ═══ B. stats download + sha256 ═══════════════════════════════════════════════
for size, (fname, sha) in U.JIT_FID_STATS.items():
    assert re.fullmatch(r"[0-9a-f]{64}", sha) and fname == f"jit_in{size}_stats.npz"
assert re.search(r"/LTH14/JiT/[0-9a-f]{40}/fid_stats/\{\}$", U.JIT_STATS_URL)

tmp = tempfile.mkdtemp()
try:
    payload = b"fake reference stats"
    real_table, real_retrieve = dict(U.JIT_FID_STATS), None
    import urllib.request
    real_retrieve = urllib.request.urlretrieve
    calls = []

    def fake_retrieve(url, dst):
        calls.append((url, dst))
        with open(dst, "wb") as f:
            f.write(payload)

    urllib.request.urlretrieve = fake_retrieve
    U.JIT_FID_STATS[256] = ("jit_in256_stats.npz", hashlib.sha256(payload).hexdigest())
    try:
        p = U.jit_fid_stats_path(256, stats_dir=tmp)
        assert p == os.path.join(tmp, "jit_in256_stats.npz") and os.path.exists(p)
        assert len(calls) == 1 and calls[0][0] == U.JIT_STATS_URL.format("jit_in256_stats.npz")
        assert calls[0][1].endswith(".part") and not os.path.exists(calls[0][1])
        U.jit_fid_stats_path(256, stats_dir=tmp)                  # cached: no 2nd download
        assert len(calls) == 1
        with open(p, "ab") as f:
            f.write(b"!")                                         # corrupt it
        try:
            U.jit_fid_stats_path(256, stats_dir=tmp)
            raise AssertionError("corrupted stats file was accepted")
        except RuntimeError as e:
            assert "sha256" in str(e)
    finally:
        urllib.request.urlretrieve = real_retrieve
        U.JIT_FID_STATS.clear()
        U.JIT_FID_STATS.update(real_table)
finally:
    shutil.rmtree(tmp)

verified = [s for s, (f, _) in U.JIT_FID_STATS.items()
            if os.path.exists(os.path.join(U.FID_STATS_DIR, f))]
for s in verified:
    U.jit_fid_stats_path(s)                                       # real file, real hash
print(f"B  stats fetched once from the pinned JiT commit via .part, sha256 checked on "
      f"every call, corruption rejected; real files verified: {verified or 'none present'}")


# ═══ C. fork detection / pinned install / re-import ══════════════════════════
pkg_root = tempfile.mkdtemp()
pkg = os.path.join(pkg_root, "torch_fidelity")
os.makedirs(pkg)
FORK_INIT = (
    "CALLS = []\n"
    "def calculate_metrics(**kw):\n"
    "    CALLS.append(kw)\n"
    "    return {'frechet_inception_distance': 1.5, 'inception_score_mean': 200.0,\n"
    "            'inception_score_std': 3.0}\n")


def write_fake(fork):
    keys = '"input1": None, "input2": None, '
    keys += '"fid_statistics_file": None, ' if fork else ""
    with open(os.path.join(pkg, "defaults.py"), "w") as f:
        f.write("DEFAULTS = {" + keys + '"cuda": True}\n')
    with open(os.path.join(pkg, "__init__.py"), "w") as f:
        f.write(FORK_INIT if fork else "STOCK = True\n")


def purge():
    for n in [n for n in sys.modules if n == "torch_fidelity" or n.startswith("torch_fidelity.")]:
        del sys.modules[n]


import importlib                                              # noqa: E402
import subprocess                                             # noqa: E402
sys.path.insert(0, pkg_root)
real_check_call = subprocess.check_call
pip_calls = []
try:
    # stock installed and already imported -> pip the pinned fork, purge, re-import
    write_fake(fork=False)
    purge()
    importlib.invalidate_caches()
    import torch_fidelity                                     # noqa: F401  (stale stock module)
    assert getattr(sys.modules["torch_fidelity"], "STOCK", False)

    def fake_pip(cmd):
        pip_calls.append(cmd)
        write_fake(fork=True)

    subprocess.check_call = fake_pip
    tf = U.ensure_jit_torch_fidelity()
    assert len(pip_calls) == 1, pip_calls
    assert pip_calls[0][1:] == ["-m", "pip", "install", "-q", "--force-reinstall",
                                "--no-deps", U.JIT_TORCH_FIDELITY], pip_calls[0]
    assert re.fullmatch(r"git\+https://github\.com/LTH14/torch-fidelity\.git@[0-9a-f]{40}",
                        U.JIT_TORCH_FIDELITY)
    assert hasattr(tf, "calculate_metrics") and not hasattr(tf, "STOCK"), "stale module kept"

    # fork already present -> no pip call
    U.ensure_jit_torch_fidelity()
    assert len(pip_calls) == 1

    # install that does not yield the fork -> hard error, no silent fallback
    write_fake(fork=False)
    purge()
    importlib.invalidate_caches()
    subprocess.check_call = lambda cmd: pip_calls.append(cmd)
    try:
        U.ensure_jit_torch_fidelity()
        raise AssertionError("missing fork did not raise")
    except RuntimeError as e:
        assert "fid_statistics_file" in str(e)
    print("C  stock torch-fidelity -> pinned `pip install --force-reinstall --no-deps` of "
          "the fork, stale module purged + re-imported; fork present -> no pip; "
          "failed install -> error")

    # ═══ D. exact engine_jit.py calculate_metrics call ════════════════════════
    write_fake(fork=True)
    purge()
    importlib.invalidate_caches()
    subprocess.check_call = lambda cmd: (_ for _ in ()).throw(AssertionError("pip called"))
    out = U.evaluate_fid_is_jit("/gen", "/stats/jit_in256_stats.npz", device="cuda")
    tf = sys.modules["torch_fidelity"]
    official = dict(input1="/gen", input2=None,
                    fid_statistics_file="/stats/jit_in256_stats.npz",
                    cuda=True, isc=True, fid=True, kid=False, prc=False, verbose=False)
    assert tf.CALLS == [official], tf.CALLS
    assert out == {"fid": 1.5, "is_mean": 200.0, "is_std": 3.0}, out
    U.evaluate_fid_is_jit("/gen", "/s.npz", device="cpu")
    assert tf.CALLS[-1]["cuda"] is False
    print("D  calculate_metrics kwargs == LTH14/JiT engine_jit.py evaluate(); "
          "result keys mapped")
finally:
    subprocess.check_call = real_check_call
    purge()
    sys.path.remove(pkg_root)
    shutil.rmtree(pkg_root)


# ═══ E. evaluate.py wiring ════════════════════════════════════════════════════
src = open(os.path.join(REPO_ROOT, "scripts", "evaluate.py"), encoding="utf-8").read()
tree = ast.parse(src)
n_default = None
for node in ast.walk(tree):
    if (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"
            and node.args and getattr(node.args[0], "value", None) == "--n_samples"):
        n_default = next(k.value for k in node.keywords if k.arg == "default")
assert isinstance(n_default, ast.Constant) and n_default.value is None
assert 'args.n_samples = s_cfg.get("n_samples") or (50000 if jit_fid else 10000)' in src
for name in ("uses_jit_fid(cfg)", "jit_fid_stats_path(", "ensure_jit_torch_fidelity()",
             "evaluate_fid_is_jit(gen_dir, jit_stats"):
    assert name in src, name
# the stats + fork are prepared before sampling starts
assert src.index("ensure_jit_torch_fidelity()") < src.index("denoiser.generate(")
print("E  evaluate.py: --n_samples default None -> 50K (ImageNet-1k) / 10K; JiT path "
      "prepared before sampling")

print("\nALL CHECKS PASSED")
