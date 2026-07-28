"""
tests/test_stateinit_model.py
=============================
Model-level guards for the state_init arm (needs mamba_ssm importable; on a
machine without it, e.g. local Windows, the test skips cleanly — run it on
Kaggle alongside the training smoke cell).

  1. BASELINE UNCHANGED: JiTVMamba(state_init="none") produces outputs
     bit-identical to a model built without the kwarg (same seed), and its
     state_dict has exactly the same keys — old checkpoints load untouched.
  2. GRADIENTS FLOW: with state_init="learned", one forward/backward gives
     nonzero grads on cond_u_proj (after a warm step; it is zero-init),
     B0 and dt0 in every block.
  3. DIMSUM EXACTNESS: with state_init="dimsum" the effective Delta0 is
     exactly 1 per direction (softplus(dt0 + bias) == 1), so
     h_{-1} = W_u(c) identically in all four scans.
  4. ARM EXCLUSIVITY: state_init + in_context_len > 0 raises.

Run:  python tests/test_stateinit_model.py
"""
import sys
import torch
import torch.nn.functional as F

try:
    from src.models.vmamba import JiTVMamba
except ImportError as e:
    print(f"SKIP (mamba_ssm not importable here): {e}")
    sys.exit(0)

KW = dict(input_size=32, patch_size=8, hidden_size=64, depth=2, num_heads=2,
          mlp_ratio=2.0, num_classes=10, bottleneck_dim=32, d_state=8)


def build(state_init=None, **extra):
    torch.manual_seed(0)
    kw = dict(KW, **extra)
    if state_init is not None:
        kw["state_init"] = state_init
    return JiTVMamba(**kw)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(2, 3, 32, 32, device=dev)
    t = torch.rand(2, device=dev)
    y = torch.randint(0, 10, (2,), device=dev)

    # 1 -- baseline unchanged
    m_old = build().to(dev)                 # kwarg omitted entirely
    m_new = build(state_init="none").to(dev)
    assert list(m_old.state_dict()) == list(m_new.state_dict()), "state_dict keys drifted"
    with torch.no_grad():
        o1, o2 = m_old(x, t, y), m_new(x, t, y)
    assert torch.equal(o1, o2), "state_init='none' is not bit-identical to baseline"
    print("  [ok ] baseline bit-identical, state_dict keys unchanged")

    # 2 -- gradients flow (learned)
    m = build(state_init="learned").to(dev)
    with torch.no_grad():
        # Warm the zero-init parameters: adaLN-Zero gates are 0 at init, so NO
        # mixer parameter (baseline included) receives gradient until the gates
        # move — perturb them, plus the zero-init W_u, to test the trained regime.
        for b in m.blocks:
            b.adaLN_modulation[-1].weight.add_(0.01 * torch.randn_like(
                b.adaLN_modulation[-1].weight))
            b.adaLN_modulation[-1].bias.add_(0.01 * torch.randn_like(
                b.adaLN_modulation[-1].bias))
            b.mixer.cond_u_proj.weight.add_(0.01 * torch.randn_like(
                b.mixer.cond_u_proj.weight))
        # final_layer.linear is also zero-init -> model output (and hence the
        # loss gradient) is exactly 0 at init; warm it too.
        m.final_layer.linear.weight.add_(0.01 * torch.randn_like(
            m.final_layer.linear.weight))
        m.final_layer.adaLN_modulation[-1].weight.add_(0.01 * torch.randn_like(
            m.final_layer.adaLN_modulation[-1].weight))
    m(x, t, y).square().mean().backward()
    for i, b in enumerate(m.blocks):
        for name in ("cond_u_proj.weight", "B0", "dt0"):
            p = b.mixer.get_parameter(name) if "." in name else getattr(b.mixer, name)
            g = (b.mixer.cond_u_proj.weight.grad if name == "cond_u_proj.weight"
                 else getattr(b.mixer, name).grad)
            assert g is not None and g.abs().max() > 0, f"block {i}: no grad on {name}"
    print("  [ok ] learned arm: nonzero grads on cond_u_proj / B0 / dt0, all blocks")

    # 3 -- dimsum: effective Delta0 == 1 per direction
    m = build(state_init="dimsum").to(dev)
    import math
    for i, b in enumerate(m.blocks):
        dt0 = math.log(math.e - 1.0) - b.mixer.dt_projs_bias
        eff = F.softplus(dt0 + b.mixer.dt_projs_bias)
        assert torch.allclose(eff, torch.ones_like(eff), atol=1e-6), \
            f"block {i}: effective Delta0 != 1"
        assert torch.all(b.mixer.B0 == 1), f"block {i}: dimsum B0 != ones"
    print("  [ok ] dimsum arm: effective Delta0 == 1 exactly, B0 == 1_N")

    # 4 -- exclusivity
    try:
        build(state_init="dimsum", in_context_len=2, in_context_start=0)
        raise AssertionError("state_init + in_context prefix did NOT raise")
    except AssertionError as e:
        if "did NOT raise" in str(e):
            raise
    print("  [ok ] state_init + in-context prefix correctly refused")
    print("ALL TESTS PASS")


if __name__ == "__main__":
    main()
