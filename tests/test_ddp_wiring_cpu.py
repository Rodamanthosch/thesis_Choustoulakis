"""
tests/test_ddp_wiring_cpu.py
============================
CPU-only verification of the multi-GPU wiring in scripts/run_experiment.py
(DDP-SYNC). DDP all-reduces gradients only for a forward that goes through
its own forward(). The old wiring wrapped the bare net but trained through
the Denoiser, which held the unwrapped net, so no gradient was ever averaged.
As in LTH14/JiT main_jit.py, DDP now wraps the whole Denoiser.

Uses PyTorch's single-process fake process group (world_size 2, no real
communication) and a DDP comm hook, which the reducer calls once per gradient
bucket it all-reduces. mamba_ssm / causal_conv1d are stubbed (references).

Checks:
  A. one process (plain `python`): wrap_ddp returns the Denoiser itself, so
     single-GPU training runs exactly the code it ran before.
  B. the old wiring all-reduced 0 times in 3 steps; the new one once per step
     (one bucket at these sizes), with gradients that match a plain backward.
  C. every model family trains 3 steps under the new DDP wiring without DDP's
     "parameter did not receive a gradient" error (find_unused_parameters is
     off, as in JiT): jit, vim, vmamba + its conditioning arms and FFNs,
     spatial_mamba, vmamba_sasf -- 16 variants. (vmamba2 needs the Triton
     SSD kernel and is not covered here.)
  D. main(): Denoiser built, wrapped, THEN init_ema (EMA starts from the
     weights DDP broadcast from rank 0); the loop trains through the wrapper;
     checkpoints and EMA use the unwrapped Denoiser; no DDP around the bare net.

Run from the repo root:
    python tests/test_ddp_wiring_cpu.py
(no PYTHONPATH needed -- the repo root is derived from this file's location)
"""
import ast
import os
import sys
import types
import warnings

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
warnings.filterwarnings("ignore")

try:
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.testing._internal.distributed.fake_pg import FakeStore
except ImportError as e:                                    # pragma: no cover
    print(f"SKIP (torch distributed / fake_pg not importable here): {e}")
    sys.exit(0)

for mod in ("yaml",):
    if mod not in sys.modules:
        try:
            __import__(mod)
        except ImportError:
            sys.modules[mod] = types.ModuleType(mod)
try:
    import tqdm                                             # noqa: F401
except ImportError:
    tq = types.ModuleType("tqdm")
    tq.tqdm = lambda it, **kw: it
    sys.modules["tqdm"] = tq


# ── stub mamba_ssm / causal_conv1d with faithful references ──────────────────
def selective_scan_ref(u, delta, A, B, C, D=None, z=None,
                       delta_bias=None, delta_softplus=False,
                       return_last_state=False):
    b, d, l = u.shape
    if delta_bias is not None:
        delta = delta + delta_bias[None, :, None].to(delta.dtype)
    if delta_softplus:
        delta = F.softplus(delta.float())
    delta, u, A = delta.float(), u.float(), A.float()
    if B.dim() == 3:
        B, C = B.unsqueeze(1), C.unsqueeze(1)
    rep = d // B.shape[1]
    Bf = B.float().repeat_interleave(rep, dim=1)
    Cf = C.float().repeat_interleave(rep, dim=1)
    h = u.new_zeros(b, d, A.shape[1])
    ys = []
    for i in range(l):
        h = torch.exp(delta[:, :, i].unsqueeze(-1) * A[None]) * h + \
            delta[:, :, i].unsqueeze(-1) * Bf[:, :, :, i] * u[:, :, i].unsqueeze(-1)
        ys.append((h * Cf[:, :, :, i]).sum(-1))
    y = torch.stack(ys, dim=-1)
    if D is not None:
        y = y + D[None, :, None].float() * u
    if z is not None:
        y = y * F.silu(z.float())
    return y


def causal_conv1d_ref(x, weight, bias=None, activation=None):
    k = weight.shape[-1]
    out = F.conv1d(F.pad(x, (k - 1, 0)), weight.unsqueeze(1), bias, groups=x.shape[1])
    return F.silu(out) if activation in ("silu", "swish") else out


if "mamba_ssm" not in sys.modules:
    stub, ops = types.ModuleType("mamba_ssm"), types.ModuleType("mamba_ssm.ops")
    iface = types.ModuleType("mamba_ssm.ops.selective_scan_interface")
    iface.selective_scan_fn = iface.selective_scan_ref = selective_scan_ref
    sys.modules.update({"mamba_ssm": stub, "mamba_ssm.ops": ops,
                        "mamba_ssm.ops.selective_scan_interface": iface})
if "causal_conv1d" not in sys.modules:
    cc = types.ModuleType("causal_conv1d")
    cc.causal_conv1d_fn = causal_conv1d_ref
    sys.modules["causal_conv1d"] = cc

from src.train import Denoiser                              # noqa: E402
from src.models.jit import JiT                              # noqa: E402
from src.models.vim import JiTViM                           # noqa: E402
from src.models.vmamba import JiTVMamba                     # noqa: E402
from src.models.spatial_mamba import JiTSpatialMamba        # noqa: E402
from src.models.vmamba_sasf import JiTVMambaSASF            # noqa: E402

# exec only wrap_ddp out of the script (its imports need torchvision)
SCRIPT = os.path.join(REPO_ROOT, "scripts", "run_experiment.py")
SRC = open(SCRIPT, encoding="utf-8").read()
fn = next(n for n in ast.parse(SRC).body
          if isinstance(n, ast.FunctionDef) and n.name == "wrap_ddp")
ns = {"DDP": DDP}
exec(compile(ast.Module([fn], []), SCRIPT, "exec"), ns)
wrap_ddp = ns["wrap_ddp"]

C = dict(input_size=16, patch_size=4, hidden_size=32, num_classes=7, bottleneck_dim=16)
NC = C["num_classes"]
x = torch.randn(4, 3, 16, 16)
y = torch.randint(0, NC, (4,))


def make_denoiser(build, seed=0):
    torch.manual_seed(seed)
    return Denoiser(net=build(), img_size=16, num_classes=NC, label_drop_prob=0.1)


def jit():
    return JiT(depth=2, num_heads=4, **C)


# ═══ A. one process: no wrapper ═══════════════════════════════════════════════
den = make_denoiser(jit)
assert wrap_ddp(den, 1, [0]) is den
print("A  world_size 1: wrap_ddp returns the Denoiser itself -> single-GPU path unchanged")

dist.init_process_group("fake", rank=0, world_size=2, store=FakeStore())


def counting_hook(state, bucket):
    state["calls"] += 1
    fut = torch.futures.Future()
    fut.set_result(bucket.buffer())          # no-op all-reduce
    return fut


def train(model, params, steps=3, hook=True):
    state = {"calls": 0}
    if hook:
        ddp = model if isinstance(model, DDP) else None
        ddp.register_comm_hook(state, counting_hook)
    for i in range(steps):
        torch.manual_seed(100 + i)
        for p in params:
            p.grad = None
        model(x, y).backward()
    return state["calls"]


# ═══ B. old wiring: never synced; new wiring: synced every step ═══════════════
torch.manual_seed(0)
old_net = jit()
old_ddp = DDP(old_net)                                   # old: DDP around the bare net
old_den = Denoiser(net=old_ddp.module, img_size=16, num_classes=NC, label_drop_prob=0.1)
state = {"calls": 0}
old_ddp.register_comm_hook(state, counting_hook)
for i in range(3):
    torch.manual_seed(100 + i)
    old_den(x, y).backward()                             # trained through the Denoiser
assert state["calls"] == 0, state

den = make_denoiser(jit)
model = wrap_ddp(den, 2)
assert isinstance(model, DDP) and model.module is den
calls = train(model, list(den.parameters()))
assert calls == 3, calls

# with the no-op hook the synced gradients equal a plain single-process backward
ref = make_denoiser(jit)
torch.manual_seed(100)
ref(x, y).backward()
den2 = make_denoiser(jit)
m2 = wrap_ddp(den2, 2)
m2.register_comm_hook({"calls": 0}, counting_hook)
torch.manual_seed(100)
m2(x, y).backward()
assert all(torch.equal(a.grad, b.grad) for a, b in zip(ref.parameters(), den2.parameters())
           if a.grad is not None)
print("B  old wiring: 0 all-reduces in 3 steps; new wiring: 3 (one per step), "
      "gradients == plain backward")


# ═══ C. every model family trains under DDP without unused-parameter errors ═══
VM = dict(depth=2, num_heads=4, d_state=8, K=4, expand=1, **C)
FAMILIES = {
    "jit":                      jit,
    "jit in-context K=4":       lambda: JiT(depth=2, num_heads=4, in_context_len=4,
                                            in_context_start=1, **C),
    "vim":                      lambda: JiTViM(depth=2, d_state=8, expand=1, **C),
    "vim in-context K=1":       lambda: JiTViM(depth=2, d_state=8, expand=1,
                                               in_context_len=1, **C),
    "vmamba":                   lambda: JiTVMamba(**VM),
    "vmamba in-context class":  lambda: JiTVMamba(in_context_len=4, in_context_content="class", **VM),
    "vmamba in-context t,y":    lambda: JiTVMamba(in_context_len=2, in_context_content="time_class", **VM),
    "vmamba state_init dimsum": lambda: JiTVMamba(state_init="dimsum", **VM),
    "vmamba state_init learned": lambda: JiTVMamba(state_init="learned", **VM),
    "vmamba ssc bc":            lambda: JiTVMamba(ssc="bc", **VM),
    "vmamba ssc abc adaln mlp": lambda: JiTVMamba(ssc="abc", adaln_cond="mlp", ssc_z_mlp=True, **VM),
    "vmamba ssc bc adaln none": lambda: JiTVMamba(ssc="bc", adaln_cond="none", **VM),
    "vmamba ffn convglu":       lambda: JiTVMamba(ffn="convglu", **VM),
    "vmamba ffn glumbconv":     lambda: JiTVMamba(ffn="glumbconv", **VM),
    "spatial_mamba gate lpu":   lambda: JiTSpatialMamba(depth=2, d_state=1, expand=1, gate=True,
                                                        lpu="zero", sasf_dilations=(1, 2, 3), **C),
    "vmamba_sasf":              lambda: JiTVMambaSASF(depth=2, d_state=1, K=4, expand=1,
                                                      sasf_dilations=(1, 2, 3), **C),
}
for name, build in FAMILIES.items():
    den = make_denoiser(build)
    model = wrap_ddp(den, 2)
    try:
        calls = train(model, list(den.parameters()))
    except RuntimeError as e:
        raise AssertionError(f"{name}: DDP failed under the new wiring:\n{e}")
    assert calls >= 3, (name, calls)
print(f"C  {len(FAMILIES)} model variants train 3 DDP steps, gradients all-reduced every "
      f"step, no unused-parameter error")


# ═══ D. wiring in main() ══════════════════════════════════════════════════════
main = SRC[SRC.index("def main():"):]


def at(s):
    i = main.find(s)
    assert i >= 0, s
    return i


assert at("denoiser = Denoiser(") < at("model = wrap_ddp(denoiser, world_size, [local_rank])") \
    < at("denoiser.init_ema()")
loop = main[at("for images, labels in pbar:"):]
assert "loss = model(images, labels)" in loop and "denoiser(images, labels)" not in loop
assert "model.train()" in main and "net.train()" not in main
assert "DDP(net" not in SRC and "raw_net" not in SRC
assert "load_checkpoint(resume, denoiser," in main and "denoiser.update_ema()" in main
n_save = main.count("save_checkpoint(")
assert n_save >= 1 and main.count("epoch, global_step, denoiser, optimizer") == n_save, \
    "every save_checkpoint must save the unwrapped Denoiser"
print("D  main(): Denoiser -> wrap_ddp -> init_ema; loop trains through the wrapper; "
      "EMA / checkpoints on the unwrapped Denoiser")

dist.destroy_process_group()
print("\nALL CHECKS PASSED")
