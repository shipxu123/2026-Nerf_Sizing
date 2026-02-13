from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml


def parse_float(value: str) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def reservoir_sample_rows(
    rows: Iterable[dict[str, str]],
    params_cols: list[str],
    metric_cols: list[str],
    max_rows: int,
    seed: int,
) -> tuple[list[list[float]], list[list[float]], int]:
    rng = np.random.default_rng(seed)
    params_samples: list[list[float]] = []
    metrics_samples: list[list[float]] = []
    skipped = 0
    count = 0

    for row in rows:
        params_vals = []
        metrics_vals = []
        valid = True

        for col in params_cols:
            val = parse_float(row.get(col))
            if val is None:
                valid = False
                break
            params_vals.append(val)

        if valid:
            for col in metric_cols:
                val = parse_float(row.get(col))
                if val is None:
                    valid = False
                    break
                metrics_vals.append(val)

        if not valid:
            skipped += 1
            continue

        count += 1
        if max_rows <= 0:
            params_samples.append(params_vals)
            metrics_samples.append(metrics_vals)
            continue

        if len(params_samples) < max_rows:
            params_samples.append(params_vals)
            metrics_samples.append(metrics_vals)
            continue

        j = int(rng.integers(0, count))
        if j < max_rows:
            params_samples[j] = params_vals
            metrics_samples[j] = metrics_vals

    return params_samples, metrics_samples, skipped


def compute_feasibility(
    metrics: np.ndarray,
    metric_names: list[str],
    feasibility_cfg: dict,
) -> np.ndarray:
    mode = feasibility_cfg.get("mode", "all_ones")
    if mode == "all_ones":
        return np.ones(len(metrics), dtype=np.float32)

    if mode != "spec_thresholds":
        raise ValueError(f"Unsupported feasibility mode: {mode}")

    specs = feasibility_cfg.get("specs", [])
    if not specs:
        return np.ones(len(metrics), dtype=np.float32)

    idx_map = {name: i for i, name in enumerate(metric_names)}
    feasibility = np.ones(len(metrics), dtype=np.float32)

    for i, row in enumerate(metrics):
        ok = True
        for spec in specs:
            name = spec["name"]
            if name not in idx_map:
                continue
            value = row[idx_map[name]]
            direction = spec.get("direction", ">=")
            target = float(spec.get("target", 0.0))

            if direction == ">=":
                if value < target:
                    ok = False
                    break
            elif direction == "<=":
                if value > target:
                    ok = False
                    break
            else:
                raise ValueError(f"Unsupported direction: {direction}")

        feasibility[i] = 1.0 if ok else 0.0

    return feasibility


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a FALCON topology dataset into NeRF-Sizing .npz format."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/falcon_cglna_conversion.yaml",
        help="Conversion config path",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output .npz path (overrides config)",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    falcon_cfg = cfg.get("falcon", {})
    dataset_root = Path(falcon_cfg.get("dataset_root", "data/FALCON/dataset"))
    topology = falcon_cfg.get("topology", "LNA/CGLNA")
    params_cols = list(falcon_cfg.get("params", []))
    metrics_map = falcon_cfg.get("metrics", {})
    pvt_vals = falcon_cfg.get("pvt", [0.0, 1.8, 27.0])
    max_rows = int(falcon_cfg.get("max_rows", 0))
    seed = int(falcon_cfg.get("seed", 42))
    feasibility_cfg = falcon_cfg.get("feasibility", {"mode": "all_ones"})

    if len(params_cols) == 0 or len(metrics_map) == 0:
        raise ValueError("Config must include non-empty params and metrics mappings.")

    if len(pvt_vals) != 3:
        raise ValueError("PVT must be a list of 3 values: [process, voltage, temperature].")

    metric_names = list(metrics_map.keys())
    metric_cols = [metrics_map[name] for name in metric_names]

    csv_path = dataset_root / topology / "dataset.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"dataset.csv not found: {csv_path}")

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in params_cols + metric_cols if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}")

        params_samples, metrics_samples, skipped = reservoir_sample_rows(
            reader,
            params_cols=params_cols,
            metric_cols=metric_cols,
            max_rows=max_rows,
            seed=seed,
        )

    if not params_samples:
        raise RuntimeError("No valid rows collected from dataset.csv")

    params = np.asarray(params_samples, dtype=np.float32)
    metrics = np.asarray(metrics_samples, dtype=np.float32)
    pvt = np.tile(np.asarray(pvt_vals, dtype=np.float32), (len(params), 1))
    feasibility = compute_feasibility(metrics, metric_names, feasibility_cfg)

    output_path = Path(args.out or cfg.get("output", "data/falcon_cglna_train_data.npz"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        params=params,
        pvt=pvt,
        metrics=metrics,
        feasibility=feasibility,
    )

    print("[INFO] Conversion complete")
    print(f"[INFO] Topology: {topology}")
    print(f"[INFO] Rows: {len(params)} (skipped {skipped})")
    print(f"[INFO] Params: {params_cols}")
    print(f"[INFO] Metrics: {metric_names} <- {metric_cols}")
    print(f"[INFO] Output: {output_path}")


if __name__ == "__main__":
    main()
