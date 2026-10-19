"""
tests/test_eval_generated_cpu.py
================================
CPU-only verification of the stale-sample fix in scripts/evaluate.py:
eval/generated/ is shared by every checkpoint and CFG value of a run, and
images were only overwritten by name, so a smaller --n_samples after a larger
one left the extra files in the folder, and FID/IS scored them too.

clear_generated / check_generated are exec'd out of evaluate.py (its module
imports need torchvision, which the CPU test env does not have).

Checks:
  A. the bug: 50 images from run 1, then run 2 writes 10 over 00000..00009 ->
     without clearing the folder still holds 50 and check_generated refuses;
     with clear_generated first, exactly run 2's 10 images remain.
  B. clear_generated removes only PNGs (any case), never other files or
     subfolders; check_generated counts every extension torch-fidelity reads
     (png / jpg / jpeg), so a stray JPEG is caught too.
  C. wiring in main(): the folder is cleared after it is created and before
     sampling, and checked after sampling and before BOTH FID paths (JiT
     stats and image folder).

Run from the repo root:
    python tests/test_eval_generated_cpu.py
(no PYTHONPATH needed -- the repo root is derived from this file's location)
"""
import ast
import os
import shutil
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(REPO_ROOT, "scripts", "evaluate.py")
SRC = open(EVAL, encoding="utf-8").read()
TREE = ast.parse(SRC)

# exec only the helpers (+ the IMAGE_EXTS constant they use)
keep = [n for n in TREE.body
        if (isinstance(n, ast.FunctionDef) and n.name in ("clear_generated", "check_generated"))
        or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "IMAGE_EXTS"
                                              for t in n.targets))]
ns = {"os": os}
exec(compile(ast.Module(keep, []), EVAL, "exec"), ns)
clear_generated, check_generated = ns["clear_generated"], ns["check_generated"]
assert ns["IMAGE_EXTS"] == (".png", ".jpg", ".jpeg")      # torch-fidelity's samples_find_ext


def write(d, n, tag):
    for i in range(n):
        with open(os.path.join(d, f"{i:05d}.png"), "w") as f:
            f.write(tag)


def raises(fn, *a):
    try:
        fn(*a)
    except RuntimeError:
        return True
    return False


tmp = tempfile.mkdtemp()
try:
    # ═══ A. the bug, and the fix ══════════════════════════════════════════════
    gen = os.path.join(tmp, "generated")
    os.makedirs(gen)
    write(gen, 50, "run1")                       # e.g. a 50K eval at CFG 2.5
    write(gen, 10, "run2")                       # then a 10K CFG sweep point, no clear
    assert len(os.listdir(gen)) == 50            # 40 run-1 images still there
    assert raises(check_generated, gen, 10), "stale images were not caught"

    assert clear_generated(gen) == 50
    write(gen, 10, "run2")
    check_generated(gen, 10)
    assert all(open(os.path.join(gen, f)).read() == "run2" for f in os.listdir(gen))
    assert clear_generated(gen) == 10 and clear_generated(gen) == 0
    print("A  50 then 10 without clearing -> refused (folder holds 50); "
          "cleared first -> exactly run 2's 10 images scored")

    # ═══ B. what is cleared / counted ═════════════════════════════════════════
    write(gen, 3, "x")
    open(os.path.join(gen, "UPPER.PNG"), "w").close()
    open(os.path.join(gen, "notes.txt"), "w").close()
    os.makedirs(os.path.join(gen, "subdir"))
    assert clear_generated(gen) == 4
    assert sorted(os.listdir(gen)) == ["notes.txt", "subdir"]
    write(gen, 5, "x")
    check_generated(gen, 5)                      # .txt and folders are not images
    open(os.path.join(gen, "stray.jpg"), "w").close()
    assert raises(check_generated, gen, 5), "stray JPEG not counted"
    os.rename(os.path.join(gen, "stray.jpg"), os.path.join(gen, "stray.jpeg"))
    assert raises(check_generated, gen, 5), "stray .jpeg not counted"
    print("B  only PNGs are deleted (other files / folders kept); png, jpg, jpeg all "
          "counted, so a stray JPEG also blocks scoring")
finally:
    shutil.rmtree(tmp)


# ═══ C. wiring in main() ══════════════════════════════════════════════════════
MAIN = SRC.index("def main():")


def at(s):
    """Position of s inside main() (the helpers' own defs come before it)."""
    i = SRC.find(s, MAIN)
    assert i >= 0, s
    return i


assert at("os.makedirs(gen_dir, exist_ok=True)") < at("clear_generated(gen_dir)") \
    < at("denoiser.generate(")
check = at("check_generated(gen_dir, args.n_samples)")
assert at("denoiser.generate(") < check
assert check < at("evaluate_fid_is_jit(gen_dir") and check < at("evaluate_fid_is(gen_dir")
print("C  main(): cleared after mkdir and before sampling; checked after sampling "
      "and before both FID paths")

print("\nALL CHECKS PASSED")
