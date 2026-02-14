# NeRF-Sizing on FALCON: Build and Run Guide

This guide shows how to run the NeRF-Sizing example using the FALCON dataset bundled under `data/FALCON`. It covers:
1. Environment setup
2. Running the minimal FALCON MLP example
3. Converting FALCON data into NeRF-Sizing `.npz`
4. Training and optimizing with NeRF-Sizing

All commands assume you are at the repo root.

## Prerequisites

- Python 3.10+
- A working PyTorch installation (CPU or CUDA)
- The FALCON dataset under `data/FALCON/dataset`

The repository already includes a full FALCON checkout under `data/FALCON`. If your dataset lives elsewhere, set `FALCON_DIR` (examples below).

## Environment Setup

You can use either the FALCON conda environment or a pip environment for NeRF-Sizing.

Option A: FALCON conda environment (recommended if you run the FALCON scripts)
```bash
conda env create -f data/FALCON/falcon.yml
conda activate falcon
pip install -e .
```

Option B: Pip environment (good for NeRF-Sizing only)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Note: The FALCON minimal example uses `pandas`. The conda env includes it. If you use pip, install `pandas` manually.

## Step 1: Run the Minimal FALCON Example (Optional Sanity Check)

The helper script `scripts/build_falcon_example.sh` runs the minimal FALCON MLP example in `data/FALCON`.
```bash
scripts/build_falcon_example.sh
```

If your FALCON repo is elsewhere:
```bash
export FALCON_DIR=/path/to/FALCON
scripts/build_falcon_example.sh
```

Common knobs for the minimal example:
```bash
python data/FALCON/scripts/example_mlp_minimal.py --per-topology 200 --max-topologies 6 --epochs 5
```

Artifacts are written to `data/FALCON/checkpoints/example_mlp_minimal`.

## Step 1b: Run the FALCON GNN Performance Predictor Demo (Stage 2)

This demo uses the Stage 2 GNN to predict analog performance metrics from circuit graphs.

One-time preprocessing (creates the GNN datasets and scalers):
```bash
cd data/FALCON
python scripts/save_gnn_data.py
cd -
```
Run the demo:
```bash
scripts/build_falcon_gnn_demo.sh --samples 3
```

Quick training on a small subset (optional):
```bash
scripts/build_falcon_gnn_demo.sh --train-quick --epochs 5 --max-train 512 --max-val 128
```

Full Stage 2 training and evaluation (uses the original FALCON scripts):
```bash
cd data/FALCON
python scripts/train_gnn.py
python evaluation/gnn_forward_eval.py
cd -
```

## Step 2: Convert FALCON to NeRF-Sizing `.npz`

The conversion config is `configs/falcon_cglna_conversion.yaml`. It expects the CGLNA topology under:
`data/FALCON/dataset/LNA/CGLNA/dataset.csv`.

Run the conversion:
```bash
python scripts/convert_falcon_to_nerf.py --config configs/falcon_cglna_conversion.yaml
```

Default output:
- `data/falcon_cglna_train_data.npz`

What the conversion does:
- Reads `dataset.csv`
- Extracts parameters: `C1 C2 Cb Ld Ls WN1`
- Extracts metrics: `PowerGain Bandwidth DCPowerConsumption`
- Samples up to `max_rows` (default 2000) using reservoir sampling

If you want all rows, set `max_rows: 0` in `configs/falcon_cglna_conversion.yaml`.

## Step 3: Train NeRF-Sizing on the FALCON Data

Use the converted `.npz` to train the NeRF-Sizing surrogate:
```bash
python scripts/train.py --config configs/falcon_cglna_nerf.yaml --data data/falcon_cglna_train_data.npz
```

Outputs:
- Checkpoint: `checkpoints/falcon_cglna/final.pt`
- Plot: `outputs/training_curves.png`

Important: The config must match the data dimensions.
- `model.input_dim_p` must be 6
- `model.output_dim_y` must be 3

The provided `configs/falcon_cglna_nerf.yaml` matches the CGLNA conversion and uses dataset-derived bounds and targets.

## Step 4: Optimize with the Trained Model

Run optimization using the saved checkpoint:
```bash
python scripts/optimize.py --config configs/falcon_cglna_nerf.yaml --checkpoint checkpoints/falcon_cglna/final.pt
```

Outputs:
- `outputs/optimization_trajectory.png` (if history is available)
- `outputs/param_comparison.png`

Note: The `configs/falcon_cglna_nerf.yaml` targets were set from dataset quantiles (Gain/BW 75th percentile, Power median). Adjust `specs` if you want stricter or looser goals.

## Step 5: Visualize Sizing Sweeps

Plot how predicted performance changes when sweeping one or more sizing parameters.

Single-parameter sweep (default uses midpoints for other params):
```bash
python scripts/plot_sizing_sweep.py --config configs/falcon_cglna_nerf.yaml \
  --checkpoint checkpoints/falcon_cglna/final.pt \
  --params C1 --num-points 80 --plot-sigma --xscale log
```

Multiple sweeps (one plot per parameter):
```bash
python scripts/plot_sizing_sweep.py --config configs/falcon_cglna_nerf.yaml \
  --checkpoint checkpoints/falcon_cglna/final.pt \
  --params C1,C2,Ld --num-points 60
```

Outputs:
- `outputs/sweep_<param>.png`
- `outputs/sweep_<param>.csv`

Notes:
- The metric labels are taken from `specs` in the config and assumed to be in the same order as the training data.
- Use `--base min|max|mid` or `--base-params` to change the fixed point for non-swept parameters.

## Step 6: Interactive Web Visualizer

The web visualizer loads the sweep CSVs and renders interactive plots in a browser.

Start a local server from the repo root:
```bash
python -m http.server 8000
```

Open the visualizer in your browser:
```text
http://localhost:8000/visualizer/
```

Load data by either:
- Selecting `outputs/sweep_*.csv` via the file picker.
- Entering a path like `outputs/sweep_C1.csv` in the path loader.

## Step 7: 3D Web Visualizer (2D Sweep)

Generate a 2D sweep CSV:
```bash
python scripts/plot_sizing_sweep_2d.py --config configs/falcon_cglna_nerf.yaml \
  --checkpoint checkpoints/falcon_cglna/final.pt \
  --params C1,Ld --num-x 50 --num-y 50 \
  --out outputs/sweep_C1_Ld.csv
```

Open the 3D visualizer:
```text
http://localhost:8000/visualizer/3d.html
```

Load the generated `outputs/sweep_C1_Ld.csv`, then pick X/Y/Z columns to render a surface, mesh, or scatter.

## Troubleshooting

- `FALCON directory not found`: set `FALCON_DIR` or ensure `data/FALCON` exists.
- `dataset.csv not found`: verify `data/FALCON/dataset/LNA/CGLNA/dataset.csv`.
- `ModuleNotFoundError: pandas`: install `pandas` or use the FALCON conda environment.
- CUDA not found: use CPU by passing `--device cpu` to `train.py` and `optimize.py`.

## Reference Files

- FALCON environment: `data/FALCON/falcon.yml`
- FALCON minimal example: `data/FALCON/scripts/example_mlp_minimal.py`
- FALCON GNN demo: `data/FALCON/scripts/example_gnn_minimal.py`
- FALCON GNN demo wrapper: `scripts/build_falcon_gnn_demo.sh`
- Conversion config: `configs/falcon_cglna_conversion.yaml`
- NeRF-Sizing configs: `configs/falcon_cglna_nerf.yaml`, `configs/default.yaml`, `configs/minimal.yaml`
