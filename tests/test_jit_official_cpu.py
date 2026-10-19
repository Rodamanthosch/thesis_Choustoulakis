"""
tests/test_jit_official_cpu.py
==============================
CPU-only verification of five behaviours aligned with LTH14/JiT:

  #1 attention kernel   src/models/jit.py scaled_dot_product_attention
  #3 CFG combination    src/train.py Denoiser._forward_sample
  #4 label dropout      src/train.py Denoiser.forward (training mode only)
  #6 weight-decay groups src/utils.py add_weight_decay (+ resume of old
                         single-group optimizer states)
  #8 non-finite loss    scripts/run_experiment.py stops before the update

The references below are verbatim transcriptions of LTH14/JiT model_jit.py,
denoiser.py and util/misc.py.

Checks:
  A. attention == JiT's scaled_dot_product_attention bit for bit (fp32), ==
     F.sdpa up to rounding, and under bf16 autocast q, k arrive in fp32 and
     the logits stay fp32 (where F.sdpa would round q, k to bf16).
  B. _forward_sample == denoiser.py bit for bit at cfg 1.0 and 2.9, for both
     interval forms; the unconditional pass now also runs at cfg 1.0; and at
     cfg 1.0 the result equals the conditional v up to rounding.
  C. label dropout: dropped in training mode only (eval sees true labels);
     the training loss and the RNG stream are bit-identical to HEAD for
     label_drop_prob 0 and 0.1.
  D. add_weight_decay: 1-D / bias params undecayed, matrices decayed, frozen
     pos_embed left out; at weight_decay 0 the two-group AdamW is bit-identical
     to the old single-group one over several steps.
  E. resume: an optimizer state saved by the old single-group AdamW loads
     into the grouped one via load_checkpoint and the next steps are
     bit-identical to continuing the old run.
  F. run_experiment.py: the non-finite check sits between the forward and
     loss.backward(), and exits.

Run from the repo root:
    python tests/test_jit_official_cpu.py
(no PYTHONPATH needed -- the repo root is derived from this file's location)
"""
import copy
import math
import os
import subprocess
import sys
import tempfile
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

try:
    import torch
    import torch.nn.functional as F
except ImportError as e:                                    # pragma: no cover
    print(f"SKIP (torch not importable here): {e}")
    sys.exit(0)

try:                                    # src.utils imports yaml for load_config only
    import yaml                                             # noqa: F401
except ImportError:
    sys.modules["yaml"] = types.ModuleType("yaml")
if "tqdm" not in sys.modules:           # src.train imports tqdm (progress bar only)
    try:
        import tqdm                                         # noqa: F401
    except ImportError:
        tq = types.ModuleType("tqdm")
        tq.tqdm = lambda it, **kw: it
        sys.modules["tqdm"] = tq

import src.models.jit as J                                  # noqa: E402
from src.models.jit import JiT                              # noqa: E402
from src.train import Denoiser                              # noqa: E402
from src.utils import (add_weight_decay, save_checkpoint,   # noqa: E402
                       load_checkpoint)


# ── verbatim LTH14/JiT references ─────────────────────────────────────────────
def jit_sdpa(query, key, value, dropout_p=0.0) -> torch.Tensor:      # model_jit.py
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_bias = torch.zeros(query.size(0), 1, L, S, dtype=query.dtype)   # .cuda() dropped (CPU)

    with torch.cuda.amp.autocast(enabled=False):
        attn_weight = query.float() @ key.float().transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


def jit_forward_sample(self, z, t, labels):                          # denoiser.py
    # conditional
    x_cond = self.net(z, t.flatten(), labels)
    v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

    # unconditional
    x_uncond = self.net(z, t.flatten(), torch.full_like(labels, self.num_classes))
    v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

    # cfg interval
    low, high = self.cfg_interval
    interval_mask = (t < high) & ((low == 0) | (t > low))
    cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

    return v_uncond + cfg_scale_interval * (v_cond - v_uncond)


def jit_add_weight_decay(model, weight_decay=0, skip_list=()):       # util/misc.py
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list or 'diffloss' in name:
            no_decay.append(param)  # no weight decay on bias, norm and diffloss
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


import warnings                                             # noqa: E402
warnings.filterwarnings("ignore", category=FutureWarning)   # torch.cuda.amp.autocast

KW = dict(input_size=16, patch_size=4, hidden_size=64, depth=4, num_heads=4,
          num_classes=7, bottleneck_dim=32)
NC = KW["num_classes"]


def make_net(seed=0, perturb=True):
    torch.manual_seed(seed)
    net = JiT(**KW)
    if perturb:                        # move off the adaLN-Zero init (zero output)
        g = torch.Generator().manual_seed(seed + 1)
        with torch.no_grad():
            for p in net.parameters():
                if p.requires_grad:
                    p.add_(torch.randn(p.shape, generator=g) * 0.05)
    return net


# ═══ A. attention kernel ═══════════════════════════════════════════════════════
torch.manual_seed(0)
q, k, v = (torch.randn(3, 4, 20, 16) for _ in range(3))
ours, ref = J.scaled_dot_product_attention(q, k, v), jit_sdpa(q, k, v)
assert torch.equal(ours, ref), (ours - ref).abs().max()
d_sdpa = (ours - F.scaled_dot_product_attention(q, k, v)).abs().max().item()
assert d_sdpa < 1e-5, d_sdpa

seen = {}
net = make_net().eval()
orig = J.scaled_dot_product_attention


def spy(query, key, value, dropout_p=0.0):
    seen.setdefault("q", query.dtype)
    seen.setdefault("k", key.dtype)
    out = orig(query, key, value, dropout_p)
    seen.setdefault("out", out.dtype)
    return out


logit_dtype = {}
real_softmax = torch.softmax
J.scaled_dot_product_attention = spy
torch.softmax = lambda x, dim: (logit_dtype.setdefault("logits", x.dtype), real_softmax(x, dim=dim))[1]
try:
    with torch.autocast("cpu", dtype=torch.bfloat16):
        net(torch.randn(2, 3, 16, 16), torch.rand(2), torch.randint(0, NC, (2,)))
finally:
    J.scaled_dot_product_attention = orig
    torch.softmax = real_softmax
assert seen["q"] == seen["k"] == torch.float32, seen
assert logit_dtype["logits"] == torch.float32, logit_dtype
print(f"A  attention == LTH14/JiT kernel bit for bit; vs F.sdpa max|diff| {d_sdpa:.1e}; "
      f"under bf16 autocast q, k, logits are fp32 (attn·V {seen['out']})")


# ═══ B. CFG combination ════════════════════════════════════════════════════════
class Counting(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net, self.calls = net, 0

    def forward(self, x, t, y):
        self.calls += 1
        return self.net(x, t, y)


bsz = 4
z = torch.randn(bsz, 3, 16, 16)
labels = torch.randint(0, NC, (bsz,))
n_cases = 0
for cfg_scale in (1.0, 2.9):
    for interval in ((0.0, 1.0), (0.1, 1.0)):
        for tv in (0.05, 0.5, 0.97):
            den = Denoiser(Counting(make_net().eval()), img_size=16, num_classes=NC,
                           cfg_scale=cfg_scale, cfg_interval=interval).eval()
            t = torch.full((bsz, 1, 1, 1), tv)
            with torch.no_grad():
                ours = den._forward_sample(z, t, labels)
                ref = jit_forward_sample(den, z, t, labels)
            assert torch.equal(ours, ref), (cfg_scale, interval, tv)
            assert den.net.calls == 4, "conditional + unconditional pass expected"
            n_cases += 1
den = Denoiser(make_net().eval(), img_size=16, num_classes=NC, cfg_scale=1.0).eval()
t = torch.full((bsz, 1, 1, 1), 0.5)
with torch.no_grad():
    v_cond = (den.net(z, t.flatten(), labels) - z) / (1.0 - t).clamp_min(den.t_eps)
    d_cond = (den._forward_sample(z, t, labels) - v_cond).abs().max().item()
assert d_cond < 1e-5, d_cond
den.steps = 3
with torch.no_grad():
    out = den.generate(labels)
assert out.shape == (bsz, 3, 16, 16) and torch.isfinite(out).all()
print(f"B  _forward_sample == LTH14/JiT denoiser.py bit for bit ({n_cases} cases, cfg 1.0/2.9, "
      f"both intervals); 2 passes at every cfg; cfg 1.0 == conditional v ({d_cond:.1e})")


# ═══ C. label dropout ══════════════════════════════════════════════════════════
head_src = subprocess.run(["git", "show", "HEAD:src/train.py"], cwd=REPO_ROOT,
                          capture_output=True, check=True).stdout.decode("utf-8")
head_mod = types.ModuleType("train_head")
exec(compile(head_src, "<HEAD:src/train.py>", "exec"), head_mod.__dict__)


class Recording(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net, self.seen = net, []

    def forward(self, x, t, y):
        self.seen.append(y.clone())
        return self.net(x, t, y)


x = torch.randn(64, 3, 16, 16)
lab = torch.randint(0, NC, (64,))
for p in (0.0, 0.1):
    rec = Recording(make_net())
    new = Denoiser(rec, img_size=16, num_classes=NC, label_drop_prob=p).train()
    old = head_mod.Denoiser(make_net(), img_size=16, num_classes=NC, label_drop_prob=p).train()
    torch.manual_seed(5)
    l_new = new(x, lab)
    s_new = torch.get_rng_state()
    torch.manual_seed(5)
    l_old = old(x, lab)
    assert torch.equal(l_new, l_old) and torch.equal(s_new, torch.get_rng_state()), p
    dropped = (rec.seen[-1] == NC).sum().item()
    assert (dropped > 0) == (p > 0), (p, dropped)
    new.eval()
    new(x, lab)
    assert torch.equal(rec.seen[-1], lab), "labels dropped in eval mode"
print("C  labels dropped in training mode only (eval sees true labels); training loss "
      "and RNG stream bit-identical to HEAD at label_drop_prob 0 and 0.1")


# ═══ D. weight-decay groups ════════════════════════════════════════════════════
net = make_net()
groups = add_weight_decay(net, 0.05)
ref_groups = jit_add_weight_decay(net, 0.05)
assert [[id(p) for p in g["params"]] for g in groups] == \
       [[id(p) for p in g["params"]] for g in ref_groups]
assert [g["weight_decay"] for g in groups] == [0.0, 0.05]
names = {id(p): n for n, p in net.named_parameters()}
no_decay = {names[id(p)] for p in groups[0]["params"]}
decay = {names[id(p)] for p in groups[1]["params"]}
assert all(n.endswith(".bias") or net.get_parameter(n).ndim == 1 for n in no_decay)
assert all(net.get_parameter(n).ndim >= 2 and not n.endswith(".bias") for n in decay)
assert "pos_embed" not in no_decay | decay
assert no_decay | decay == {n for n, p in net.named_parameters() if p.requires_grad}


def train_steps(net, opt, n, seed):
    g = torch.Generator().manual_seed(seed)
    for _ in range(n):
        xb = torch.randn(4, 3, 16, 16, generator=g)
        tb = torch.rand(4, generator=g)
        yb = torch.randint(0, NC, (4,), generator=g)
        opt.zero_grad(set_to_none=True)
        net(xb, tb, yb).pow(2).mean().backward()
        opt.step()


a, b = make_net(), make_net()
opt_a = torch.optim.AdamW(a.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.0)
opt_b = torch.optim.AdamW(add_weight_decay(b, 0.0), lr=1e-3, betas=(0.9, 0.95))
train_steps(a, opt_a, 5, seed=11)
train_steps(b, opt_b, 5, seed=11)
assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
print(f"D  groups == LTH14/JiT add_weight_decay ({len(no_decay)} undecayed, {len(decay)} "
      f"decayed, pos_embed excluded); weight_decay 0: 2-group AdamW == 1-group "
      f"bit for bit over 5 steps")


# ═══ E. resume an old single-group optimizer state ════════════════════════════
old_net = make_net()
old_den = Denoiser(old_net, img_size=16, num_classes=NC)
old_den.init_ema()
old_opt = torch.optim.AdamW(old_net.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.0)
train_steps(old_net, old_opt, 3, seed=21)
tmp = tempfile.mkdtemp()
ckpt = os.path.join(tmp, "old.pt")
save_checkpoint(ckpt, 1, 3, old_den, old_opt, [0.1], 0.1)

cont_net = copy.deepcopy(old_net)                     # the old run, continued
cont_opt = torch.optim.AdamW(cont_net.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.0)
cont_opt.load_state_dict(old_opt.state_dict())
train_steps(cont_net, cont_opt, 3, seed=22)

res_net = make_net(seed=99)                           # resumed with the new optimizer
res_den = Denoiser(res_net, img_size=16, num_classes=NC)
res_den.init_ema()
res_opt = torch.optim.AdamW(add_weight_decay(res_net, 0.0), lr=1e-3, betas=(0.9, 0.95))
load_checkpoint(ckpt, res_den, optimizer=res_opt)
assert len(res_opt.state_dict()["param_groups"]) == 2
train_steps(res_net, res_opt, 3, seed=22)
assert all(torch.equal(p, q) for p, q in zip(cont_net.parameters(), res_net.parameters()))
os.remove(ckpt)
os.rmdir(tmp)
print("E  old single-group checkpoint -> load_checkpoint into the grouped AdamW; next "
      "3 steps bit-identical to continuing the old run")


# ═══ F. non-finite loss stops before the update ════════════════════════════════
src = open(os.path.join(REPO_ROOT, "scripts", "run_experiment.py"), encoding="utf-8").read()
loop = src[src.index("for images, labels in pbar:"):]
i_fwd = loop.index("loss = model(images, labels)")
i_chk = loop.index("if not math.isfinite(loss_value):")
i_exit = loop.index("sys.exit(1)")
i_bwd = loop.index("loss.backward()")
i_step = loop.index("optimizer.step()")
assert i_fwd < i_chk < i_exit < i_bwd < i_step
assert "loss = model(images, labels)\n                loss.backward()" not in loop
assert "add_weight_decay(net, " in src and "net.parameters(), lr=peak_lr" not in src
print("F  run_experiment.py: forward -> isfinite check -> exit, before backward/step; "
      "optimizer built from add_weight_decay")

print("\nALL CHECKS PASSED")
