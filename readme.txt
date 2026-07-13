Here's the full step-by-step to actually get that script running.

## Step 1 — Check prerequisites

- **Python**: 3.10–3.13 (3.10–3.12 recommended if you're also using NIM tooling)
- **GPU**: An NVIDIA GPU with CUDA is strongly recommended (inference will run but be very slow on CPU). Check with:
```bash
nvidia-smi
```
- **Disk space**: A few GB free — model checkpoints (FCN3 etc.) get downloaded on first run.

## Step 2 — Set up an isolated environment

Using `venv` (simplest, no extra tools needed):
```bash
python3 -m venv e2studio-env
source e2studio-env/bin/activate      # on Windows: e2studio-env\Scripts\activate
pip install --upgrade pip
```
(NVIDIA's own docs recommend `uv` for a smoother dependency resolution experience, but plain `pip`/`venv` works fine for a first run.)

## Step 3 — Install PyTorch first

Earth2Studio rides on top of PyTorch, so get that installed and GPU-aware *before* installing Earth2Studio itself. Go to https://pytorch.org/get-started/locally/, pick your CUDA version, and run the command it gives you, e.g.:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```
Verify:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
should print `True` if the GPU is visible.

## Step 4 — Install Earth2Studio with the model extra you need

For the FCN3 model in the script:
```bash
pip install "earth2studio[fcn3] @ git+https://github.com/NVIDIA/earth2studio.git"
```
(If you'd rather use the original slide's `FCN` model, swap the extra: `earth2studio[fcn]`.)

## Step 5 — Handle NGC auth quirks (only if you hit an error)

Model checkpoints download via NGC. Most public models work with anonymous/guest access, but if you already have an NGC API key configured on your machine, it can sometimes cause `ValueError: Invalid org` errors on *public* model downloads. If that happens:
```bash
mv ~/.ngc/config ~/.ngc/config.bak   # temporarily disable it
# or
unset NGC_CLI_API_KEY
```

## Step 6 — Save and run the script

Save the file I generated (`earth2studio_forecast_example.py`) locally, then:
```bash
python earth2studio_forecast_example.py
```
What happens:
1. Downloads the FCN3 checkpoint (first run only — cached after that).
2. Fetches GFS initial-condition data for `2024-01-01T00:00:00` over the network.
3. Runs 20 autoregressive steps on the GPU.
4. Writes results to `outputs/fcn3_forecast.zarr` and prints the output data tree.

## Step 7 — Inspect the output

```python
import xarray as xr
ds = xr.open_zarr("outputs/fcn3_forecast.zarr")
print(ds)
ds["t2m"].isel(time=0, lead_time=5).plot()   # e.g. 2m temperature at step 5
```

## Step 8 — Common troubleshooting

| Symptom | Likely fix |
|---|---|
| `ModuleNotFoundError` for a model-specific package | You installed the base package without the model extra — reinstall with `earth2studio[fcn3]` (or whichever model) |
| Build errors on `torch-harmonics` / `earth2grid` | Needs system build tools (`python3-dev`, `cmake`); see the troubleshooting docs |
| `ValueError: Invalid org/team` | NGC credential conflict — see Step 5 |
| Very slow inference | Confirm `torch.cuda.is_available()` is `True`; you're likely running on CPU |

Want me to also walk through swapping in a different model (e.g. GraphCastOperational or AIFS) or setting up an ensemble forecast instead of a single deterministic run?
