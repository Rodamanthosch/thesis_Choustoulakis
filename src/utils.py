"""
Shared utilities: config loading, FID helpers, throughput measurement.
"""
import yaml
import time
import torch


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    """Load a YAML config file and return as dict."""
    with open(path) as f:
        return yaml.safe_load(f)


# ── Throughput ────────────────────────────────────────────────────────────────

@torch.no_grad()
def measure_throughput(model, input_shape, device, n_warmup=5, n_iters=50):
    """
    Measure inference throughput (images/sec).

    Args:
        model: PyTorch model
        input_shape: tuple (B, C, H, W)
        device: torch device
    Returns:
        float: images per second
    """
    model.eval()
    B, C, H, W = input_shape
    x = torch.randn(B, C, H, W, device=device)
    t = torch.rand(B, device=device)
    y = torch.randint(0, 10, (B,), device=device)

    for _ in range(n_warmup):
        model(x, t, y)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(n_iters):
        model(x, t, y)
    torch.cuda.synchronize()

    return (n_iters * B) / (time.time() - start)


# ── Logging ───────────────────────────────────────────────────────────────────

class MetricLogger:
    """Simple in-memory metric logger; saves to JSON."""
    import json, os

    def __init__(self):
        self.history = []

    def log(self, step: int, **kwargs):
        self.history.append({"step": step, **kwargs})

    def save(self, path: str):
        import json, os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"Metrics saved → {path}")
