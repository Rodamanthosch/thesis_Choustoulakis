"""
tests/test_ssc_model.py
=======================
Model-level guards for the SSC arm (skips cleanly without mamba_ssm).

  1. BASELINE UNCHANGED: ssc="none" bit-identical to a model built without
     the kwarg; state_dict keys unchanged (old checkpoints load as-is).
  2. INIT SEMANTICS: with ssc="bc" at init, W_B = W_C = 0, so the model
     equals a Mamba-3-style STATIC-bias model — verified by comparing
     against manually zeroing the condition projections.
  3. GRADIENTS FLOW ("abc"): bias_B/C, cond_B/C_proj, gate_A_w, gate_A_b all
     receive nonzero grads in every block (after warming zero-init params).
  4. GATE RANGE: at init the gate is sigmoid(4.0) ~ 0.982 for every sample.
  5. ARM EXCLUSIVITY: ssc + state_init, and ssc + in-context, both raise.

Run:  python tests/test_ssc_model.py
"""
import sys
import torch

try:
    from src.models.vmamba import JiTVMamba
except ImportError as e:
    print(f"SKIP (mamba_ssm not importable here): {e}")
    sys.exit(0)

KW = dict(input_size=32, patch_size=8, hidden_size=64, depth=2, num_heads=2,
          mlp_ratio=2.0, num_classes=10, bottleneck_dim=32, d_state=8)


def build(**extra):
    torch.manual_seed(0)
    return JiTVMamba(**KW, **extra)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(2, 3, 32, 32, device=dev)
    t = torch.rand(2, device=dev)
    y = torch.randint(0, 10, (2,), device=dev)

    # 1 -- baseline unchanged
    m_old, m_new = build().to(dev), build(ssc="none").to(dev)
    assert list(m_old.state_dict()) == list(m_new.state_dict())
    with torch.no_grad():
        assert torch.equal(m_old(x, t, y), m_new(x, t, y)), "ssc='none' not bit-identical"
    print("  [ok ] baseline bit-identical, state_dict keys unchanged")

    # 2 -- init semantics: bc at init == static-bias-only model
    m = build(ssc="bc").to(dev)
    with torch.no_grad():
        o1 = m(x, t, y)
        for b in m.blocks:  # explicitly zero the (already zero) condition projections
            b.mixer.cond_B_proj.weight.zero_(); b.mixer.cond_C_proj.weight.zero_()
        o2 = m(x, t, y)
    assert torch.equal(o1, o2), "condition projections not zero at init"
    print("  [ok ] bc arm at init == Mamba-3-style static-bias model")

    # 3 -- gradients flow (abc)
    m = build(ssc="abc").to(dev)
    with torch.no_grad():
        for b in m.blocks:
            b.adaLN_modulation[-1].weight.add_(0.01 * torch.randn_like(b.adaLN_modulation[-1].weight))
            b.adaLN_modulation[-1].bias.add_(0.01 * torch.randn_like(b.adaLN_modulation[-1].bias))
            b.mixer.cond_B_proj.weight.add_(0.01 * torch.randn_like(b.mixer.cond_B_proj.weight))
            b.mixer.cond_C_proj.weight.add_(0.01 * torch.randn_like(b.mixer.cond_C_proj.weight))
        m.final_layer.linear.weight.add_(0.01 * torch.randn_like(m.final_layer.linear.weight))
        m.final_layer.adaLN_modulation[-1].weight.add_(
            0.01 * torch.randn_like(m.final_layer.adaLN_modulation[-1].weight))
    m(x, t, y).square().mean().backward()
    for i, b in enumerate(m.blocks):
        for name in ("bias_B", "bias_C", "gate_A_w", "gate_A_b"):
            g = getattr(b.mixer, name).grad
            assert g is not None and g.abs().max() > 0, f"block {i}: no grad on {name}"
        for name in ("cond_B_proj", "cond_C_proj"):
            g = getattr(b.mixer, name).weight.grad
            assert g is not None and g.abs().max() > 0, f"block {i}: no grad on {name}"
    print("  [ok ] abc arm: nonzero grads on all SSC parameters, all blocks")

    # 4 -- gate value at init
    m = build(ssc="abc").to(dev)
    c = torch.randn(5, KW["hidden_size"], device=dev)
    g = torch.sigmoid(torch.einsum("bd,kd->bk", c, m.blocks[0].mixer.gate_A_w) +
                      m.blocks[0].mixer.gate_A_b).clamp(min=0.1)
    assert torch.allclose(g, torch.full_like(g, torch.sigmoid(torch.tensor(4.0)).item()), atol=1e-6)
    print(f"  [ok ] abc gate at init = sigmoid(4.0) = {g[0,0].item():.4f} for all samples")

    # 5 -- exclusivity
    for bad in (dict(ssc="bc", state_init="dimsum"),
                dict(ssc="bc", in_context_len=2, in_context_start=0)):
        try:
            build(**bad)
            raise RuntimeError(f"{bad} did NOT raise")
        except AssertionError:
            pass
    print("  [ok ] ssc + other arms correctly refused")
    print("ALL TESTS PASS")


if __name__ == "__main__":
    main()
