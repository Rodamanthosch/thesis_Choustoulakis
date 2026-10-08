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

## JiT-S2-Spatial-Mamba (`model: spatial_mamba`)

`src/models/spatial_mamba.py` is a **separate model** (`vmamba.py` untouched):
the JiT scaffold with Spatial-Mamba's Structure-aware SSM (Xiao et al., ICLR
2025) as the mixer — ONE row-major scan, the hidden states fused over the H×W
grid by dilated depthwise 3×3 convs (SASF), then `y = C·h + D·u`. The states
come out of the **stock** kernel: with `d_state = 1` (asserted), calling
`selective_scan_fn` with `C = 1, D = None` returns `x_t` exactly, so no forked
CUDA op is needed. Baseline conditioning only (adaLN-Zero); the in-context /
state-init / SSC arms are not ported.

| key | values | meaning |
|---|---|---|
| `gate` | `true` (default) \| `false` | `LN(y)·SiLU(z)` as in the paper \| no z branch (VMamba `v05_noz`) |
| `sasf_dilations` | list, default `[1, 3, 5]` | paper's at 16×16; Tiny-ImageNet (8×8) configs use `[1, 2, 3]` |
| `lpu` | `none` (default) \| `zero` \| `paper` | residual DWConv3×3 before mixer and FFN; `zero` keeps blocks identity at init |
| `expand`, `ffn` | as `vmamba` | |

| Tiny-IN config | params |
|---|---|
| `jit-s2-vmamba-dstate1` (SS2D, `d_state: 1`, config only) | 30.20M |
| `jit-s2-spatial-gate` (expand 1, gate, lpu zero) | 31.46M |
| `jit-s2-spatial-nogate` (expand 1, no gate, lpu zero) | 29.69M |

`tests/test_spatial_mamba_cpu.py` proves the mixer against a naive transcription
of the paper's Eq. (2)–(3), `StateFusion` against the official module, and (check
J) that the copied block/model scaffold is bit-identical to `JiTVMamba`'s;
`scripts/check_spatial_cuda.py` is the GPU check.

When adding a new arm, keep the "off" setting byte-identical to the previous
model and prove it in a CPU test under `tests/` (see
`tests/test_incontext_row_cpu.py` check A, which transplants weights from
`git show HEAD:` and asserts `max|diff| == 0`).

## Wiring

`build_model` is duplicated in `scripts/run_experiment.py` and
`scripts/evaluate.py`. These have drifted apart before — **update both** whenever
a model flag is added (for `vmamba` and `spatial_mamba` alike), or evaluation
will silently build a different model than training did. `scripts/profile_model.py` has no vmamba conditioning pass-through
at all; complexity numbers come from `evaluate.py::build_model` via
`notebooks/tiny_imagenet/jit-s2-vmamba-tinyin-complexity.ipynb`.

## Tests

CPU tests stub `mamba_ssm` with a reference selective scan, so they run without a
GPU. Run them from the repo root with `PYTHONPATH=.` — without it,
`from src.models.vmamba import ...` fails and the tests' `try/except` misreports
it as a missing `mamba_ssm`.
