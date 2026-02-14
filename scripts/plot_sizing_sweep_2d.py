from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 2D sweep CSV for 3D visualization.")
    parser.add_argument("--config", type=str, default="configs/falcon_cglna_nerf.yaml", help="Config path")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/falcon_cglna/final.pt", help="Model checkpoint")
    parser.add_argument("--params", type=str, default="", help="Two param names to sweep, e.g., C1,Ld")
    parser.add_argument("--num-x", type=int, default=40, help="Number of X sweep points")
    parser.add_argument("--num-y", type=int, default=40, help="Number of Y sweep points")
    parser.add_argument("--pvt", type=str, default="", help="Override PVT as 'process,voltage,temp'")
    parser.add_argument("--pvt-index", type=int, default=None, help="Index into PVT lists in config")
    parser.add_argument("--base", type=str, default="mid", choices=["min", "mid", "max"], help="Base point for other params")
    parser.add_argument("--base-params", type=str, default="", help="Override base params as comma list")
    parser.add_argument("--out", type=str, default="outputs/sweep_2d.csv", help="Output CSV path")
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
    sweep_params = parse_csv_strings(args.params) if args.params else []
    if len(sweep_params) != 2:
        raise ValueError("Provide exactly two params via --params, e.g., --params C1,Ld")

    if sweep_params[0] not in param_names or sweep_params[1] not in param_names:
        raise ValueError(f"Unknown params. Available: {param_names}")

    idx_x = param_names.index(sweep_params[0])
    idx_y = param_names.index(sweep_params[1])

    base_override = parse_csv_floats(args.base_params) if args.base_params else None
    base_params = build_base_params(param_bounds, args.base, base_override)

    pvt_override = parse_csv_floats(args.pvt) if args.pvt else None
    pvt = get_pvt_from_config(cfg, pvt_override, args.pvt_index)

    model, norm_p, norm_v, norm_y = load_model_and_norms(cfg, checkpoint, device)

    lower_x = float(param_bounds[idx_x]["min"])
    upper_x = float(param_bounds[idx_x]["max"])
    lower_y = float(param_bounds[idx_y]["min"])
    upper_y = float(param_bounds[idx_y]["max"])

    values_x = np.linspace(lower_x, upper_x, args.num_x, dtype=np.float32)
    values_y = np.linspace(lower_y, upper_y, args.num_y, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(values_x, values_y, indexing="xy")

    total_points = grid_x.size
    params = np.tile(base_params, (total_points, 1))
    params[:, idx_x] = grid_x.reshape(-1)
    params[:, idx_y] = grid_y.reshape(-1)

    p_tensor = torch.tensor(params, dtype=torch.float32, device=device)
    v_tensor = torch.tensor(np.tile(pvt, (total_points, 1)), dtype=torch.float32, device=device)

    with torch.no_grad():
        p_norm = norm_p.normalize(p_tensor)
        v_norm = norm_v.normalize(v_tensor)
        sigma_hat, y_hat_norm = model(p_norm, v_norm)
        y_hat = norm_y.denormalize(y_hat_norm)

    sigma = sigma_hat.cpu().numpy().reshape(-1)
    y = y_hat.cpu().numpy()

    metric_names = [spec["name"] for spec in cfg.get("specs", [])]
    if not metric_names:
        metric_names = [f"metric_{i}" for i in range(y.shape[1])]

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    headers = [sweep_params[0], sweep_params[1], "sigma", *metric_names[: y.shape[1]]]
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i in range(total_points):
            row = [float(params[i, idx_x]), float(params[i, idx_y]), float(sigma[i])]
            row.extend(float(v) for v in y[i].tolist())
            writer.writerow(row)

    print(f"[INFO] Saved 2D sweep CSV to {output_path}")


if __name__ == "__main__":
    main()
