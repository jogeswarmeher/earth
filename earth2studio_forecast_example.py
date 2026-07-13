"""
AI Weather Forecast with Earth2Studio
Updated to match the current (v0.14+) Earth2Studio API.

Install (only the FCN3 extra, e.g.):
    pip install "earth2studio[fcn3] @ git+https://github.com/NVIDIA/earth2studio.git"
"""

import os

from earth2studio.data import GFS
from earth2studio.io import ZarrBackend
from earth2studio.models.px import FCN3          # updated FourCastNet (successor to FCN)
from earth2studio.run import deterministic as run

# Make sure there's somewhere to write the output store
os.makedirs("outputs", exist_ok=True)

# 1. Prognostic model: load the pretrained checkpoint from NGC
package = FCN3.load_default_package()
model = FCN3.load_model(package)

# 2. Data source: pull initial conditions from NOAA GFS
data = GFS()

# 3. IO backend: persist the forecast to a Zarr store on disk
io = ZarrBackend("outputs/fcn3_forecast.zarr")

# 4. Run inference: 20 steps (6 hr/step for FCN-family models => 5 days)
nsteps = 20
io = run(["2024-01-01T00:00:00"], nsteps, model, data, io)

print(io.root.tree())
