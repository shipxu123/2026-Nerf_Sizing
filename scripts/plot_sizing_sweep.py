from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Iterable

# Avoid writing Matplotlib cache to a quota-limited home directory.
_repo_root = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(_repo_root / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nerf_sizing.model.circuit_field_net import CircuitFieldNet
from nerf_sizing.utils.config import load_config
from nerf_sizing.utils.normalization import Normalizer


def parse_csv_floats(text: str) -> list[float]:
    values = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(float(chunk))
    return values


def parse_csv_strings(text: str) -> list[str]:
    values = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if chunk:
            values.append(chunk)
    return values


def get_pvt_from_config(cfg: dict, pvt_override: list[float] | None, pvt_index: int | None) -> np.ndarray:
    if pvt_override is not None:
        if len(pvt_override) != 3:
            raise ValueError("PVT override must have 3 values: process, voltage, temperature.")
        return np.asarray(pvt_override, dtype=np.float32)

    pvt_cfg = cfg.get("pvt", {})
    process = pvt_cfg.get("process", [0.0])
    voltage = pvt_cfg.get("voltage", [1.8])
    temperature = pvt_cfg.get("temperature", [27.0])

    def pick(values: Iterable[float]) -> float:
        values = list(values)
        if not values:
            raise ValueError("Empty PVT list in config.")
        if pvt_index is None:
            return float(values[0])
        if pvt_index < 0 or pvt_index >= len(values):
            raise ValueError(f"PVT index {pvt_index} out of range for {values}.")
        return float(values[pvt_index])

    return np.asarray([pick(process), pick(voltage), pick(temperature)], dtype=np.float32)


def load_model_and_norms(cfg: dict, checkpoint: Path, device: str) -> tuple[CircuitFieldNet, Normalizer, Normalizer, Normalizer]:
    model = CircuitFieldNet.from_config(cfg["model"]).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    norm_p = Normalizer()
    norm_v = Normalizer()
    norm_y = Normalizer()
    norm_p.load_state_dict(ckpt["norm_p"])
    norm_v.load_state_dict(ckpt["norm_v"])
    norm_y.load_state_dict(ckpt["norm_y"])

    return model, norm_p, norm_v, norm_y


def build_base_params(param_bounds: list[dict], mode: str, override: list[float] | None) -> np.ndarray:
    if override is not None:
        if len(override) != len(param_bounds):
            raise ValueError("Base override length must match param count.")
        return np.asarray(override, dtype=np.float32)

    lower = np.asarray([p["min"] for p in param_bounds], dtype=np.float32)
    upper = np.asarray([p["max"] for p in param_bounds], dtype=np.float32)

    if mode == "min":
        return lower
    if mode == "max":
        return upper
    return (lower + upper) / 2.0


def sweep_param(
    param_values: np.ndarray,
    base_params: np.ndarray,
    param_index: int,
    pvt: np.ndarray,
    model: CircuitFieldNet,
    norm_p: Normalizer,
    norm_v: Normalizer,
    norm_y: Normalizer,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    params = np.tile(base_params, (len(param_values), 1))
    params[:, param_index] = param_values

    p_tensor = torch.tensor(params, dtype=torch.float32, device=device)
    v_tensor = torch.tensor(np.tile(pvt, (len(param_values), 1)), dtype=torch.float32, device=device)

    with torch.no_grad():
        p_norm = norm_p.normalize(p_tensor)
        v_norm = norm_v.normalize(v_tensor)
        sigma_hat, y_hat_norm = model(p_norm, v_norm)
        y_hat = norm_y.denormalize(y_hat_norm)

    return param_values, sigma_hat.cpu().numpy().squeeze(), y_hat.cpu().numpy()


def plot_sweep(
    x: np.ndarray,
    sigma: np.ndarray,
    y: np.ndarray,
    metric_names: list[str],
    specs: dict[str, dict],
    param_name: str,
    xscale: str,
    yscale: str,
    out_path: Path,
    plot_sigma: bool,
) -> None:
    num_metrics = y.shape[1]
    rows = num_metrics + (1 if plot_sigma else 0)
    fig, axes = plt.subplots(rows, 1, figsize=(8, max(3, 2.6 * rows)), sharex=True)
    if rows == 1:
        axes = [axes]

    for i in range(num_metrics):
        ax = axes[i]
        ax.plot(x, y[:, i], linewidth=1.6)
        name = metric_names[i] if i < len(metric_names) else f"metric_{i}"
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        ax.set_yscale(yscale)

        spec = specs.get(name)
        if spec:
            target = spec.get("target")
            direction = spec.get("direction", ">=")
            target_numeric = None
            if isinstance(target, (int, float, np.floating)):
                target_numeric = float(target)
            else:
                try:
                    target_numeric = float(target)
                except (TypeError, ValueError):
                    target_numeric = None

            if target_numeric is not None:
                ax.axhline(target_numeric, color="tab:red", linestyle="--", linewidth=1.2)
                ax.set_title(f"{name} (target {direction} {target_numeric:g})")
            else:
                ax.set_title(f"{name} (target {direction} {target})")
        else:
            ax.set_title(name)

    if plot_sigma:
        ax = axes[-1]
        ax.plot(x, sigma, color="tab:purple", linewidth=1.6)
        ax.set_ylabel("sigma")
        ax.set_title("Feasibility (sigma)")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    axes[-1].set_xlabel(param_name)
    axes[-1].set_xscale(xscale)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_csv(
    path: Path,
    param_name: str,
    x: np.ndarray,
    sigma: np.ndarray,
    y: np.ndarray,
    metric_names: list[str],
) -> None:
    headers = [param_name, "sigma"]
    headers.extend(metric_names[: y.shape[1]])
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i in range(len(x)):
            row = [float(x[i]), float(sigma[i])]
            row.extend(float(v) for v in y[i].tolist())
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot sizing sweep curves using a trained NeRF-Sizing model.")
    parser.add_argument("--config", type=str, default="configs/falcon_cglna_nerf.yaml", help="Config path")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/falcon_cglna/final.pt", help="Model checkpoint")
    parser.add_argument("--params", type=str, default="", help="Comma-separated param names to sweep (default: first)")
    parser.add_argument("--num-points", type=int, default=50, help="Number of sweep points")
    parser.add_argument("--pvt", type=str, default="", help="Override PVT as 'process,voltage,temp'")
    parser.add_argument("--pvt-index", type=int, default=None, help="Index into PVT lists in config")
    parser.add_argument("--base", type=str, default="mid", choices=["min", "mid", "max"], help="Base point for other params")
    parser.add_argument("--base-params", type=str, default="", help="Override base params as comma list")
    parser.add_argument("--xscale", type=str, default="linear", choices=["linear", "log"], help="X axis scale")
    parser.add_argument("--yscale", type=str, default="linear", choices=["linear", "log"], help="Y axis scale")
    parser.add_argument("--plot-sigma", action="store_true", help="Plot feasibility sigma as the last panel")
    parser.add_argument("--out-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Device: cpu/cuda/auto")
    args = parser.parse_args()

    cfg = load_config(args.config)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    param_bounds = cfg.get("params", [])
    if not param_bounds:
        raise ValueError("Config is missing params.")

    param_names = [p["name"] for p in param_bounds]
    sweep_params = parse_csv_strings(args.params) if args.params else [param_names[0]]

    base_override = parse_csv_floats(args.base_params) if args.base_params else None
    base_params = build_base_params(param_bounds, args.base, base_override)

    pvt_override = parse_csv_floats(args.pvt) if args.pvt else None
    pvt = get_pvt_from_config(cfg, pvt_override, args.pvt_index)

    model, norm_p, norm_v, norm_y = load_model_and_norms(cfg, checkpoint, device)

    specs = {spec["name"]: spec for spec in cfg.get("specs", [])}
    metric_names = [spec["name"] for spec in cfg.get("specs", [])]
    if not metric_names:
        metric_names = [f"metric_{i}" for i in range(cfg["model"]["output_dim_y"])]

    out_dir = Path(args.out_dir)

    for param_name in sweep_params:
        if param_name not in param_names:
            raise ValueError(f"Unknown param '{param_name}'. Available: {param_names}")

        idx = param_names.index(param_name)
        lower = float(param_bounds[idx]["min"])
        upper = float(param_bounds[idx]["max"])
        values = np.linspace(lower, upper, args.num_points, dtype=np.float32)

        x, sigma, y = sweep_param(
            values,
            base_params,
            idx,
            pvt,
            model,
            norm_p,
            norm_v,
            norm_y,
            device,
        )

        plot_path = out_dir / f"sweep_{param_name}.png"
        plot_sweep(
            x,
            sigma,
            y,
            metric_names,
            specs,
            param_name,
            args.xscale,
            args.yscale,
            plot_path,
            plot_sigma=args.plot_sigma,
        )

        csv_path = out_dir / f"sweep_{param_name}.csv"
        save_csv(csv_path, param_name, x, sigma, y, metric_names)

        print(f"[INFO] Saved {plot_path} and {csv_path}")


if __name__ == "__main__":
    main()
