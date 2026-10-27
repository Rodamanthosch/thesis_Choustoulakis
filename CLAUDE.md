# CLAUDE.md

Project conventions for Claude Code sessions in this repository.

## Git conventions

- Do **not** add `Co-Authored-By: Claude ...` trailers to commit messages.
- Do **not** add the "Generated with Claude Code" footer to pull request descriptions.

Commits are authored by Rodamanthos Choustoulakis only. GitHub reads
`Co-Authored-By` trailers and lists the co-author on the repository's
contributors page, which is not wanted here.

## Conditioning arms (JiT-S2-VMamba)

Every conditioning route is selected purely by config flags and the arms are
**mutually exclusive** (asserted in `JiTVMamba.__init__`): enable at most one of
`in_context_len > 0`, `state_init != "none"`, `ssc != "none"` per run. All
defaults reproduce the adaLN-Zero baseline byte-for-byte.

| Arm | Flags under `model:` |
|---|---|
| baseline (adaLN-Zero only) | all defaults |
| state-init | `state_init: dimsum` \| `learned` |
| SSC | `ssc: static` \| `bc` \| `abc` |
| in-context prefix | `in_context_len: K`, `in_context_content: class` \| `time_class`, `in_context_start: 4` |
| in-context row | the above **plus** `in_context_layout: row` |

`in_context_layout` defaults to `prefix`; `row` requires `in_context_len` to be a
multiple of `grid_size` (and exactly `2 * grid_size` for `time_class`).

`adaln_cond` is a **modifier of the SSC arm**, not an arm of its own (it does not
enter the `n_arms` count): it selects which residual branches adaLN still carries
`(t, y)` into. SSC can only modulate `B`/`C`(/`A`) *inside* the mixer, so
`mlp` is the honest "SSC replaces adaLN" comparison and `none` additionally
strips the FFN and the output head.

| `adaln_cond` | mixer branch | FFN branch + `FinalLayer` | S/12 params |
|---|---|---|---|
| `full` (default, `true`) | adaLN-Zero | adaLN-Zero | 31.62M (ssc-bc) |
| `mlp` | zero-init static bias | adaLN-Zero | 26.61M |
| `none` (`false`) | zero-init static bias | zero-init static bias | 21.01M |

Anything but `full` requires `ssc != "none"` and forbids the in-context prefix.
`ssc_z_mlp: true` (legal only when `adaln_cond != "full"`) gives SSC DiM-2's
dedicated `z = MLP(t, c)`; `z` goes to the **SSC path only**, while any surviving
adaLN keeps consuming the raw `c`. `none` keeps the parameter names the
notebooks' `install_noadaln.sh` used (`blocks.*.adaLN_bias`,
`final_layer.adaLN_bias`), so checkpoints from those runs still load.

## FFN variant (`ffn`)

`ffn` swaps the block FFN and is **orthogonal** to every conditioning arm (it
does not enter `n_arms`; combine it with any arm by adding one config key).
Both conv FFNs keep SwiGLU's param-matched width `int(mlp_ratio * D * 2/3)`
and use SiLU; `u` is the modulated `norm2(x)`, DW a depthwise 3×3 over the
H×W token grid.

| `ffn` | maths | per block (D=384) | Tiny-IN S/12 |
|---|---|---|---|
| `swiglu` (default) | `W2(SiLU(g)·v)`, `[g,v]=W1 u` | 1,182,080 | 31.03M |
| `convglu` (TransNeXt, GELU→SiLU) | `W2(SiLU(DW(g))·v)` | +10,240 | 31.15M |
| `glumbconv` (SANA Mix-FFN) | `s=DW(SiLU(W1 u))`, `[a,g]=s`, `W3(a·SiLU(g))`, no out bias | +20,096 | 31.27M |

In-context prefix tokens (the first `N - H·W` of `x`, for both layouts) are
not on the grid: they see only the conv's centre tap, so the FFN never mixes
prefix and image tokens. `tests/test_ffn_cpu.py` proves `swiglu` is
bit-identical to HEAD and that both conv FFNs match verbatim transcriptions of
the TransNeXt / SANA modules; `scripts/check_ffn_cuda.py` is the GPU check.

## Norm layer (`norm`)

`norm: derf` replaces every block's `norm1` / `norm2` and the head's
`norm_final` RMSNorm with Dynamic erf (Chen et al., "Stronger
Normalization-Free Transformers", CVPR 2026),
`Derf(x) = weight · erf(alpha·x + shift) + bias`: `alpha`, `shift` scalars
(init `derf_alpha_init` = 0.5, 0), `weight`, `bias` per channel (init 1, 0), computed
in fp32 like RMSNorm. These are the paper's pre-attention, pre-FFN and final
positions; SS2D's post-scan `out_norm` and Spatial-Mamba's `LN(y)` are not
pre-norms, are untested in the paper and stay LayerNorm. Like `ffn` it is
**orthogonal** to every arm (not in `n_arms`) and exists for `vmamba`,
`vmamba_sasf`, `spatial_mamba` and `vmamba2`. The swap
(`src/primitives.py::convert_rmsnorm_to_derf`) runs after `initialize_weights`
and draws no random numbers, so every other weight equals the RMSNorm model's
at the same seed; adaLN-Zero is kept (the paper's DiT recipe drops it for
point-wise norms, but its released code keeps it on DiT-B's blocks).

| `norm` | per swapped norm | Tiny-IN `jit-s2-vmamba-derf` |
|---|---|---|
| `rmsnorm` (default) | `weight` | 31,031,616 (baseline) |
| `derf` | + `bias`, `alpha`, `shift` (D + 2) | 31,041,266 (+25 × 386) |

`tests/test_derf_cpu.py` proves `rmsnorm` is bit-identical to HEAD for all four
models, that `Derf` matches a verbatim copy of the official `Dynamic_erf`, and
that exactly 2·depth + 1 norms are swapped with every baseline weight kept.

## JiT-S2-Spatial-Mamba (`model: spatial_mamba`)

`src/models/spatial_mamba.py` is a **separate model** (`vmamba.py` untouched):
the JiT scaffold with Spatial-Mamba's Structure-aware SSM (Xiao et al., ICLR
2025) as the mixer — ONE row-major scan, the hidden states fused over the H×W
grid by dilated depthwise 3×3 convs (SASF), then `y = C·h + D·u`. The states
come out of the **stock** kernel: with `d_state = 1`, calling
`selective_scan_fn` with `C = 1, D = None` returns `x_t` exactly, so no forked
CUDA op is needed. `d_state > 1` (an extension; the paper defines N = 1) runs
the scan over `N·d_inner` single-state channels (u, Δ repeated; `A[c, n]`;
B-group n carries `B_t[n]`), which is the N-state Mamba-1 recurrence exactly;
SASF fuses all N·d_inner state maps and `y[c] = Σ_n C[n]·h[c, n] + D[c]·u[c]`.
The d_state 1 forward is the original code path, unchanged. Conditioning:
adaLN-Zero, plus optional **in-context class tokens placed inside the scan**
(below); the state-init / SSC arms and `adaln_cond` are not ported.

In-context class tokens follow the VMamba arm's persist & update scheme
(`y_emb + slot`, prepended once at `in_context_start`, stripped after the last
block), but `in_context_positions` puts token e **just before grid cell
(row, col)** of the one row-major scan instead of all at its head (`null` =
all before (0, 0), the DiM prefix). Tokens share `in_proj`, skip the mixer's
DWConv, get only the centre tap of the LPU / conv FFNs, and are **not fused**
by SASF (grid states only); their output is `C·h + D·u` of their own state.
Because SASF reads states up to max(dilation) rows/columns away in every
direction, tokens mid-row reach the cells on both sides: on 8×8 with
`[1, 2, 3]`, cells (0,3), (3,3), (7,3) put every image token within 2 scan
steps (mean 1.19) of a class token, vs 37 / 12.25 for one head token.

| key | values | meaning |
|---|---|---|
| `gate` | `true` (default) \| `false` | `LN(y)·SiLU(z)` as in the paper \| no z branch (VMamba `v05_noz`) |
| `sasf_dilations` | list, default `[1, 3, 5]` | paper's at 16×16; Tiny-ImageNet (8×8) configs use `[1, 2, 3]` |
| `lpu` | `none` (default) \| `zero` \| `paper` | residual DWConv3×3 before mixer and FFN; `zero` keeps blocks identity at init |
| `d_state` | default 1 | >1: N states per channel (see above) |
| `sasf_state_kernels` | `shared` (default) \| `per_state` | d_state > 1 only: one kernel set per channel for all its N maps \| one per (channel, state) map (+27·d_inner·(N−1) params per block); same compute |
| `grad_checkpoint` | `false` (default) \| `true` | recompute each mixer in the backward pass (same maths, bit-identical); on in the d_state-16 configs: ≈24 → 2.5 GiB activations at batch 128 |
| `in_context_len` | default 0 | number of class tokens; 0 = adaLN-Zero only, byte-for-byte the baseline |
| `in_context_start`, `in_context_content` | default 0, `class` | block the tokens enter at; only `class` is ported |
| `in_context_positions` | `null` (default) \| `[[row, col], ...]` | one cell per token, scanned just before it; `null` = head prefix |
| `expand`, `ffn` | as `vmamba` | |

| Tiny-IN config | params |
|---|---|
| `jit-s2-vmamba-dstate1` (SS2D, `d_state: 1`, config only) | 30.20M |
| `jit-s2-spatial-gate` (expand 1, gate, lpu zero) | 31.46M |
| `jit-s2-spatial-nogate` (expand 1, no gate, lpu zero) | 29.69M |
| `jit-s2-spatial-nogate-nolpu` (as above, `lpu: none`: LPU ablation vs `jit-s2-vmamba-sasf`) | 29,593,956 |
| `jit-s2-spatial-nogate-dstate16-shared` (no gate, d_state 16, shared kernels, lpu zero) | 29.89M |
| `jit-s2-spatial-nogate-dstate16-perstate` (no gate, d_state 16, per-state kernels, lpu zero) | 31.76M |
| `jit-s2-spatial-nogate-dstate4-shared` (no gate, d_state 4, shared kernels, lpu zero, no checkpointing: ≈8.3 GiB activations fits a T4) | 29.73M |
| `jit-s2-spatial-nogate-dstate4-perstate` (no gate, d_state 4, per-state kernels, lpu zero, no checkpointing) | 30.10M |
| `jit-s2-spatial-nogate-incontext-class3` (no gate, lpu zero, 3 class tokens before (0,3) (3,3) (7,3) from block 4) | 29,687,268 |

`tests/test_spatial_mamba_cpu.py` proves the mixer against a naive transcription
of the paper's Eq. (2)–(3), `StateFusion` against the official module, and (check
J) that the copied block/model scaffold is bit-identical to `JiTVMamba`'s;
`scripts/check_spatial_cuda.py` is the GPU check. `tests/test_spatial_dstate16_cpu.py`
proves d_state 1 is bit-identical to HEAD, the N·d_inner-channel scan equals the
N-state recurrence, the N = 16 mixer equals a naive transcription for both
`sasf_state_kernels`, and `shared` == `per_state` with tied kernels.
`tests/test_spatial_incontext_cpu.py` proves tokens-off is bit-identical to HEAD,
the mixer with tokens equals a naive float64 transcription, the placement is
causal, and recomputes the distance figures from the module's scan order;
`scripts/check_spatial_incontext_cuda.py` is the GPU check (exactness, bf16
fwd+bwd, it/s and peak memory vs the no-gate run).

## JiT-S2-VMamba + SASF (`model: vmamba_sasf`)

`src/models/vmamba_sasf.py` is a **separate model** built from `vmamba.py`
itself (which it does not modify): Spatial-Mamba's SASF added **inside each of
SS2D's four scan directions** at `d_state = 1` (asserted) — Spatial-Mamba's
Appendix F "VMamba + SASF". Per direction k: states read with `C = 1, D = None`,
un-permuted to the grid (`cross_unscan`), fused with **that direction's own**
kernels and α (`DirectionalStateFusion`), then `y = Σ_k (C^k·h^k + D^k·u)`,
which is exactly CrossMerge. `SS2DSASF` subclasses `SS2D` (same init and
parameter names, + `state_fusion`); `JiTVMambaSASF` subclasses
`JiTVMamba(d_state=1)` and swaps each block's SS2D for an SS2DSASF carrying the
same weights, so with identity fusion it **is** JiT-VMamba at d_state 1. No LPU,
no gate; baseline conditioning only.

| key | values | meaning |
|---|---|---|
| `sasf_dilations` | list, default `[1, 3, 5]` | Tiny-ImageNet config uses `[1, 2, 3]` |
| `d_state` | must be 1 | |
| `expand`, `K`, `ffn` | as `vmamba` | |

Tiny-IN `jit-s2-vmamba-sasf`: 30,699,984 params (SS2D 380,928 + SASF 41,484 per
block). `tests/test_vmamba_sasf_cpu.py` proves `cross_unscan`, the per-direction
fusion against 4 independent `StateFusion`s, identity fusion == `SS2D` and ==
the whole `JiTVMamba(d_state=1)`, and the mixer against a naive per-direction
transcription. `scripts/check_sasf_cuda.py` is the GPU check for this model and
the d_state 16 Spatial runs (exactness, bf16 fwd+bwd, it/s and peak memory).

When adding a new arm, keep the "off" setting byte-identical to the previous
model and prove it in a CPU test under `tests/` (see
`tests/test_incontext_row_cpu.py` check A, which transplants weights from
`git show HEAD:` and asserts `max|diff| == 0`).

## Wiring

`build_model` is duplicated in `scripts/run_experiment.py` and
`scripts/evaluate.py`. These have drifted apart before — **update both** whenever
a model flag is added (for `vmamba`, `spatial_mamba`, `vmamba_sasf` and `vmamba2` alike), or evaluation
will silently build a different model than training did. `scripts/profile_model.py` has no vmamba conditioning pass-through
at all; complexity numbers come from `evaluate.py::build_model` via
`notebooks/tiny_imagenet/jit-s2-vmamba-tinyin-complexity.ipynb`.

## Tests

CPU tests stub `mamba_ssm` with a reference selective scan, so they run without a
GPU. Run them from the repo root with `PYTHONPATH=.` — without it,
`from src.models.vmamba import ...` fails and the tests' `try/except` misreports
it as a missing `mamba_ssm`.
