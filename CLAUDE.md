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
| in-context diagonal | `in_context_len: K`, `in_context_content: class`, `in_context_start: 4`, `in_context_layout: diagonal` |

`in_context_layout` defaults to `prefix`; `row` requires `in_context_len` to be a
multiple of `grid_size` (and exactly `2 * grid_size` for `time_class`).

`diagonal` ties token e to grid cell (e·s, e·s), s = `grid_size // in_context_len`
(so `in_context_len` must divide `grid_size`; `class` content only). Unlike
`prefix` / `row`, which insert after `cross_scan` and are never flipped, its
position is permuted **with the image** (one gather per path,
`vmamba.py::diagonal_scan_index`): just before its cell in paths 0/2, just after
it in the reversed paths 1/3. Same params as `prefix` at the same length; model
token layout, persist & update and the strip are unchanged.

| 8×8, 8 tokens, per block (DWConv aside) | cells each token reads | cells each token writes to |
|---|---|---|
| `prefix` | 0 | 64 |
| `row` | 0, 28, 48, 60, 64, 64, 64, 64 | 64, 64, 64, 64, 64, 60, 48, 28 |
| `diagonal` | 64 | 64 |

The diagonal is our own placement: Mamba-R (arXiv 2405.14858) spaces registers
evenly in the flattened 1-D sequence only, and its 2-D shape depends on the grid
and register count (diagonal only for 14×14 with 12). What is taken from it is
the flip of the registers with the sequence. Tiny-IN
`jit-s2-vmamba-incontext-diagonal-class`: 31,034,688 params, blocks 4–11.
`tests/test_incontext_diagonal_cpu.py` proves `prefix` / `row` are bit-identical
to HEAD, the gather with 0 tokens is `cross_scan` / `cross_merge` bit for bit, the
placement against an independent derivation from `cross_scan`'s orders, the
round trip, and the read/write table per cell (row and prefix included);
`scripts/check_incontext_diagonal_cuda.py` is the GPU check.

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

## Feature-level RoPE (`feature_rope`)

`src/feature_rope.py`, for `vmamba` and `spatial_mamba` only. A fixed rotation of
the mixer's **image features**, `rotated_features[position] = R(position) @
features[position]`, turning each adjacent channel pair by its own angle:
`[x0, x1] -> [x0 cos a - x1 sin a, x0 sin a + x1 cos a]`. No parameters, buffers
or RNG. It is a positional modifier (not in `n_arms`) added on top of the
additive sin-cos embedding, which stays.

| `model:` key | values | meaning |
|---|---|---|
| `feature_rope` | `none` (default) \| `1d` \| `2d_axial` | off \| row-major index \| row / column axes |
| `feature_rope_theta` | default `10000.0` | finite, positive; only read when active |

With D = `d_inner`, cell (r, c) of an H×W grid, integer coordinates from 0 and no
normalisation or interpolation:

| mode | needs | frequencies | angle of pair j |
|---|---|---|---|
| `1d` | D even | `omega[j] = theta ** (-2j/D)`, j < D/2 | `(r*W + c) * omega[j]` |
| `2d_axial` | 4 divides D | `omega[j] = theta ** (-4j/D)`, j < D/4 | first D/2 channels `r * omega[j]`, last D/2 `c * omega[j]` |

`2d_axial` is JiT's `VisionRotaryEmbeddingFast` construction taken over D feature
channels instead of head dimensions (bit-identical on an 8×8 grid). Angles and
tables are computed in fp32 and cached per (mode, theta, D, H, W, device, dtype)
at module level, outside inference mode and outside every `state_dict`.

Insertion point: `in_proj → DWConv2D → SiLU → FEATURE RoPE`, on the canonical
grid, in both mixers. In SS2D that is before `cross_scan`, so positions are never
recomputed per path; the rotated features feed x_proj (Δ, B, C), the scan and the
D skip term. In Spatial-Mamba they feed x_proj, the scan and `y = C·h + D·u`; the
gate branch `z` is not rotated. Condition tokens (prefix, row, diagonal, Spatial
positions) join after the rotation, so they are not rotated and do not shift an
image coordinate.

**This is an adaptation inspired by RRG-Mamba (Hou et al., IJCAI 2025), not a
reproduction.** Its Figure 2 places RoPE after DWConv + SiLU and before SS2D, but
Eq. (10) writes a query-key inner product without saying how that becomes the
sequence SS2D consumes, and the GDLM code was unavailable. Rotating the features
is our reading: there is no Q/K projection or similarity matrix, and the SSM
state, B and C are not rotated directly. RoFormer's relative-position guarantee
is a property of attention's Q-K product; do not claim it for the Mamba block.
DPB, the paper's extra gates and convolutions and its report generator are not
included. `vmamba_sasf`, `vmamba2` and `vim` are not ported.

| Tiny-IN config under `configs/tiny_imagenet/` | params (same as its source) |
|---|---|
| `jit-s2-vmamba-feature-rope-{1d,2d-axial}` (d_state 16) | 31,031,616 |
| `jit-s2-spatial-nogate-feature-rope-{1d,2d-axial}` (d_state 1) | 29,686,116 |

`tests/test_feature_rope_cpu.py` proves `none` is bit-identical to the pre-edit
models (mixers directly, and the full model with every weight perturbed), both
rotations against an independent loop-based reference on a rectangular grid, the
placement (pre-edit mixer + rotation spliced after its SiLU == new mixer; rotated
cells at their exact position in all four paths; tokens untouched), finite
forward + backward, strict loading both ways and the train / eval / profile
builders on the four configs. It uses the stubbed scan, so it does not exercise
the CUDA kernel; `scripts/check_feature_rope_cuda.py --pair vmamba|spatial --mode 1d|2d_axial`
is the GPU check, run by the four `notebooks/tiny_imagenet/*-feature-rope-*.ipynb`. Set `FEATURE_ROPE_REFERENCE_DIR` to compare against a saved copy
of the pre-edit tree; otherwise it uses the last commit before the feature.

## B/C RoPE (`bc_rope`) and A tied in pairs (`a_tie`)

`vmamba` only. RoPE on the scan's **key B and query C**, where it is actually
relative (unlike `feature_rope`). The scan reads
`y_i = Σ_j C_iᵀ(Π decay)B_j Δ_j x_j`; rotating B and C in adjacent d_state pairs
by their cell's 2-D position gives `(R_i C_i)ᵀ decay R_j B_j = C_iᵀ decay R(φ_j − φ_i) B_j`
**only if both states of every pair share A**. Mamba-1's A is free per
(channel, state), so `bc_rope` requires `a_tie: pairs`, which needs its own
control run. Both are modifiers (not in `n_arms`).

| `model:` key | values | meaning |
|---|---|---|
| `a_tie` | `none` (default) \| `pairs` | `A_logs` (K, d_inner, N/2), `repeat_interleave(2)` in forward; init = the baseline's even states, A = −[1,1,3,3,…,N−1,N−1]; needs even d_state |
| `bc_rope` | `none` (default) \| `2d_axial` | rotate B, C after x_proj's split (and after `ssm_norm`); pairs j < N/4 by row, the rest by column, `omega[j] = theta ** (-4j/N)` (feature_rope's 2d_axial over d_state); tables permuted by `cross_scan`, so a cell keeps its angle in all four paths |
| `bc_rope_theta` | default `10.0` | at d_state 16: 1, .56, .32, .18 rad/cell (θ = 1e4 would leave half the frequencies near zero on 8×8) |

`bc_rope` needs `a_tie: pairs`, d_state divisible by 4, K = 4, and is asserted
off with in-context tokens, `state_init` and SSC (no angle is defined for a
token / the virtual position; SSC offsets B, C). It combines with `ssm_norm`
and `feature_rope`. Δ, B and C still come from unrotated features, the stock
kernel is unchanged, angles are computed in the work dtype (≥ fp32), and B, C
reach the scan in their original dtype. Exactness is for the B/C product only:
the model still sees absolute position (sin-cos, DWConv padding, scan order).
`vmamba2` would need no tying (scalar A per head) but is not ported;
`spatial_mamba` (SASF fuses states across cells before C reads them) and
`vmamba_sasf` (d_state 1) are not either. `src/internals.py` reads `A_logs`
directly and does not expand a tied one.

| Tiny-IN config under `configs/tiny_imagenet/` | params |
|---|---|
| `jit-s2-vmamba-atie` (control) | 30,884,160 (−147,456) |
| `jit-s2-vmamba-bc-rope-2d-axial` | 30,884,160 (same as the control) |

`tests/test_bc_rope_cpu.py` proves both flags off are bit-identical to commit
`118940d` (pinned; `BC_ROPE_REFERENCE=<rev>` overrides) for every arm and
`vmamba_sasf`, the tables against an independent float64 derivation, the mixer
against a naive float64 transcription (also with `ssm_norm: bcdt`), that a
constant coordinate shift leaves the output unchanged with A tied (~1e-16) and
moves it with A free, the params / init / dtypes / guards, and the three
builders on both configs. `scripts/check_bc_rope_cuda.py --config atie|bc-rope`
is the GPU check, run by `notebooks/tiny_imagenet/jit-s2-vmamba-tinyin-{atie,bc-rope-2d-axial}.ipynb`.

## SSM-parameter norm (`ssm_norm`)

Jamba's RMSNorm on the input-dependent SSM parameters (Lieber et al., arXiv
2403.19887, §6.4; transformers' `JambaMambaMixer` `dt/b/c_layernorm`), applied
right after `x_proj`'s split: `dt_r` (dt_rank wide, **before** `dt_proj`; bias +
softplus stay in the scan) and B, C (d_state wide). Each is `JambaRMSNorm`:
learnable per-channel gain init 1, eps 1e-6, statistic in at least fp32
(`src/primitives.py::ssm_rms_norm`; float64 stays float64, a bf16 input stays
bf16 so the CUDA scan sees u's dtype). B and C are the scan's key and query
(`y_i = Σ_j C_iᵀ(Π decay)B_j Δ_j x_j`), so `bcdt` is QK-norm's Mamba analogue.
Like `ffn` / `norm` it is **orthogonal** (not in `n_arms`), but asserted off
with `ssc` (offsets B/C after the split) and `state_init` (`B0` bypasses
`x_proj`): untested pairings. Plain `selective_scan_fn` + autograd, so the
backward is exact (unlike Falcon Mamba's fused `mamba_inner_fn` route, which
treats the norm as identity in the backward pass).

| model | `ssm_norm` | gains per block | Tiny-IN config, params |
|---|---|---|---|
| `vmamba` | `none` (default) \| `dt` \| `bcdt` (needs d_state ≥ 2) | `dt` K·R, `bcdt` K·(R + 2N), one set per direction, `blocks.*.mixer.{dt,B,C}_norm_weight` (flat 1-D → no weight decay) | `jit-s2-vmamba-ssmnorm-bcdt` 31,034,304 (+2,688) |
| `spatial_mamba` | `none` (default) \| `dt` | R, `blocks.*.mixer.dt_norm_weight` | `jit-s2-spatial-nogate-ssmnorm-dt` 29,686,404 (+288) |

`bcdt` is rejected at d_state 1 and on `spatial_mamba`: an RMSNorm over one
B / C value is its sign. `vmamba_sasf` / `vmamba2` are not ported.
`tests/test_ssm_norm_cpu.py` proves `none` is bit-identical to HEAD for both
models and the arms next to it, `ssm_rms_norm` == a verbatim `JambaRMSNorm`,
the SS2D (`bcdt`, `dt`) and Spatial (`dt`) mixers == naive float64
transcriptions, scale invariance in the normalised x_proj rows, the gains
(1-D, ones, no decay, gradients), unchanged scan dtypes under bf16, the guards
and the wiring; `scripts/check_ssm_norm_cuda.py --pair vmamba|spatial` is the
GPU check (kernel vs `selective_scan_ref`, bf16 fwd+bwd, it/s, peak memory).

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

## JiT-S2-VMamba2 (`model: vmamba2`)

`src/models/vmamba2.py` is a **separate model** (`vmamba.py` untouched): SS2D
with a **Mamba-2 (SSD) core**, following VMamba's own Mamba-2 variant (`SS2Dm0`,
`forward_type="m0_noz"`). Everything around the SSM is the baseline SS2D
(in_proj, DWConv3×3 + SiLU, 4-direction cross-scan/merge, LayerNorm, out_proj,
no gate); only the core changes: Δ and A one **per head** instead of Δ per
channel and A per (channel, state). Heads as VMamba m0: `ceil(D/16)` per
direction (24 × headdim 16 at D=384), B/C one group per direction. Init is the
Mamba-2 module's (A ~ U[1,16], Δ log-uniform [1e-3, 1e-1]); D per channel as
in the baseline. Kernel: `mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined`
(mamba-ssm 2.2.4). Baseline conditioning only; the arms are not ported.

| key | values | meaning |
|---|---|---|
| `d_state` | default 16 | 16 = the baseline's total state (24 × 16 × 16 = 384 × 16 per direction) |
| `nheads` | `null` (default) \| int | heads per direction; `null` → `ceil(D/16)` (VMamba m0) |
| `chunk_size` | default 64 | SSD chunk (VMamba m0); L = 64 on Tiny-IN is one chunk |
| `force_fp32` | `false` (default) \| `true` | kernel inputs in fp32 (VMamba m0's flag) — for GPUs whose Triton can't do bf16 `tl.dot` |
| `expand`, `K`, `ffn` | as `vmamba` | |

Tiny-IN `jit-s2-vmamba2`: 30,278,208 params (mixer 450,048 → 387,264 per block
vs the Mamba-1 baseline). `tests/test_vmamba2_cpu.py` proves the SSD stub
against an independent dual form, that the mixer **equals the Mamba-1 SS2D with
Δ and A tied per head** (check C, float64), and (check G) that the copied
scaffold is bit-identical to `JiTVMamba`'s. `scripts/check_vmamba2_cuda.py` is
the GPU check **and** hardware probe: it reports whether the Triton kernel runs
on the GPU in bf16 / fp32 and prints which `force_fp32` to use. Triton
officially targets Ampere+; run it on Kaggle T4 before training there.

When adding a new arm, keep the "off" setting byte-identical to the previous
model and prove it in a CPU test under `tests/` (see
`tests/test_incontext_row_cpu.py` check A, which transplants weights from
`git show HEAD:` and asserts `max|diff| == 0`).

## Wiring

`build_model` is duplicated in `scripts/run_experiment.py` and
`scripts/evaluate.py`. These have drifted apart before — **update both** whenever
a model flag is added (for `vmamba`, `spatial_mamba`, `vmamba_sasf` and `vmamba2` alike), or evaluation
will silently build a different model than training did. `scripts/profile_model.py` passes only `feature_rope`, `a_tie` and `bc_rope` through for vmamba and no conditioning flags
at all, and does not build `spatial_mamba`; complexity numbers come from `evaluate.py::build_model` via
`notebooks/tiny_imagenet/jit-s2-vmamba-tinyin-complexity.ipynb`.

## Tests

CPU tests stub `mamba_ssm` with a reference selective scan, so they run without a
GPU. Run them from the repo root with `PYTHONPATH=.` — without it,
`from src.models.vmamba import ...` fails and the tests' `try/except` misreports
it as a missing `mamba_ssm`.
