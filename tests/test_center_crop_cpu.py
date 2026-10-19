"""
tests/test_center_crop_cpu.py
=============================
CPU-only verification of the ImageNet crop (src/utils.py center_crop_arr /
CenterCropArr), which now replaces T.Resize + T.CenterCrop in
run_experiment.py::build_dataset and evaluate.py's real-image set.
Needs only PIL + numpy (no torchvision, no GPU).

Checks:
  A. center_crop_arr == a verbatim transcription of LTH14/JiT util/crop.py
     (pixel-exact) on 40 random sizes: portrait / landscape / square, odd and
     even, upsampling (short side < size), and BOX-halving cases (short side
     >= 2x and >= 4x size), for image_size 256 and 512, RGB.
  B. output is image_size x image_size RGB, and the BOX halving really runs
     (its result differs from a single BICUBIC resize on large inputs).
  C. size-preserving input is untouched: 64x64 -> 64 (Tiny-ImageNet) and
     32x32 -> 32 return the same pixels, so those datasets are unchanged.
  D. CenterCropArr == center_crop_arr, and it pickles (DataLoader workers).

Run from the repo root:
    python tests/test_center_crop_cpu.py
(no PYTHONPATH needed -- the repo root is derived from this file's location)
"""
import os
import pickle
import sys
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np                                          # noqa: E402
from PIL import Image                                       # noqa: E402

try:                                    # src.utils imports yaml for load_config only
    import yaml                                             # noqa: F401
except ImportError:
    sys.modules["yaml"] = types.ModuleType("yaml")

from src.utils import center_crop_arr, CenterCropArr       # noqa: E402


# ── verbatim transcription of LTH14/JiT util/crop.py ─────────────────────────
def center_crop_arr_jit(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


rng = np.random.default_rng(0)


def rand_image(w, h):
    """Smooth-ish RGB photo stand-in: a low-res noise field upsampled + noise."""
    base = rng.integers(0, 256, (max(h // 16, 2), max(w // 16, 2), 3), dtype=np.uint8)
    img = Image.fromarray(base).resize((w, h), Image.BILINEAR)
    arr = np.array(img).astype(np.int16) + rng.integers(-20, 21, (h, w, 3))
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8), "RGB")


def same(a, b):
    return a.size == b.size and a.mode == b.mode and np.array_equal(np.array(a), np.array(b))


# ═══ A. pixel-exact vs the JiT transcription ════════════════════════════════
fixed = [(500, 375), (375, 500), (256, 256), (257, 300), (300, 257), (200, 150),
         (512, 512), (513, 700), (1024, 768), (2048, 1100), (1100, 2048), (777, 1023)]
sizes = fixed + [tuple(int(v) for v in rng.integers(64, 1600, 2)) for _ in range(28)]
n = 0
for image_size in (256, 512):
    for w, h in sizes:
        img = rand_image(w, h)
        assert same(center_crop_arr(img, image_size), center_crop_arr_jit(img, image_size)), \
            f"mismatch vs JiT crop at {w}x{h} -> {image_size}"
        n += 1
print(f"A  center_crop_arr pixel-exact vs LTH14/JiT util/crop.py on {n} cases")


# ═══ B. shape / mode, and the BOX halving is live ═════════════════════════════
for w, h in sizes:
    out = center_crop_arr(rand_image(w, h), 256)
    assert out.size == (256, 256) and out.mode == "RGB", (w, h, out.size, out.mode)
big = rand_image(1100, 2048)                    # short side 1100 >= 4 * 256: halves twice
scale = 256 / 1100
one_shot = big.resize((round(1100 * scale), round(2048 * scale)), Image.BICUBIC)
arr = np.array(one_shot)
cy, cx = (arr.shape[0] - 256) // 2, (arr.shape[1] - 256) // 2
one_shot = Image.fromarray(arr[cy:cy + 256, cx:cx + 256])
assert not same(center_crop_arr(big, 256), one_shot), "BOX halving had no effect"
print("B  output always 256x256 RGB; BOX halving changes large inputs vs one BICUBIC resize")


# ═══ C. size-preserving input untouched (Tiny-ImageNet 64, CIFAR 32) ═════════
for s in (64, 32):
    for _ in range(5):
        img = rand_image(s, s)
        assert same(center_crop_arr(img, s), img), f"{s}x{s} -> {s} changed pixels"
print("C  64x64 -> 64 and 32x32 -> 32 return identical pixels (Tiny-ImageNet unchanged)")


# ═══ D. transform wrapper + pickling ═════════════════════════════════════════
t = pickle.loads(pickle.dumps(CenterCropArr(256)))
for w, h in fixed:
    img = rand_image(w, h)
    assert same(t(img), center_crop_arr(img, 256))
print(f"D  CenterCropArr == center_crop_arr after a pickle round trip ({t!r})")

print("\nALL CHECKS PASSED")
