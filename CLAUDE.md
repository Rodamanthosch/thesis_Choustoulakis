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

When adding a new arm, keep the "off" setting byte-identical to the previous
model and prove it in a CPU test under `tests/` (see
`tests/test_incontext_row_cpu.py` check A, which transplants weights from
`git show HEAD:` and asserts `max|diff| == 0`).

## Wiring

`build_model` is duplicated in `scripts/run_experiment.py` and
`scripts/evaluate.py`. These have drifted apart before — **update both** whenever
a model flag is added, or evaluation will silently build a different model than
training did. `scripts/profile_model.py` has no vmamba conditioning pass-through
at all; complexity numbers come from `evaluate.py::build_model` via
`notebooks/tiny_imagenet/jit-s2-vmamba-tinyin-complexity.ipynb`.

## Tests

CPU tests stub `mamba_ssm` with a reference selective scan, so they run without a
GPU. Run them from the repo root with `PYTHONPATH=.` — without it,
`from src.models.vmamba import ...` fails and the tests' `try/except` misreports
it as a missing `mamba_ssm`.
