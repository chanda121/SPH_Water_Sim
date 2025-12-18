"""
Visualize SPH simulation output files in time order.
Loops through frames showing particle positions colored by height (rho).
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from pysph.solver.utils import get_files, load

# Get output directory from command line or use default
outdir = sys.argv[1] if len(sys.argv) > 1 else "2D_Water_Sim_output"
files = get_files(outdir)

if not files:
    raise SystemExit(f"No output files in {outdir}")

# Sort files by simulation time 
times = []
for fp in files:
    d = load(fp)
    t = float(d["solver_data"]["t"])
    times.append((t, fp))
times.sort(key=lambda x: x[0])
files = [fp for _, fp in times]

print(f"Found {len(files)} output files")
print(f"Time range: {times[0][0]:.4f} to {times[-1][0]:.4f}")

# --- Find global bounds across ALL frames for consistent display ---
xmin = ymin = np.inf
xmax = ymax = -np.inf
vmin = np.inf  # min height/rho
vmax = -np.inf  # max height/rho

for fpath in files:
    dic = load(fpath)
    pa = dic["arrays"]["fluid"]
    x, y = pa.x, pa.y
    H = pa.height if hasattr(pa, "height") else pa.rho

    xmin = min(xmin, float(x.min()))
    xmax = max(xmax, float(x.max()))
    ymin = min(ymin, float(y.min()))
    ymax = max(ymax, float(y.max()))
    vmin = min(vmin, float(H.min()))
    vmax = max(vmax, float(H.max()))

print(f"Position bounds: x=[{xmin:.2f}, {xmax:.2f}], y=[{ymin:.2f}, {ymax:.2f}]")
print(f"Height bounds: [{vmin:.3f}, {vmax:.3f}]")

# Add padding around bounds
padx = 0.2 * (xmax - xmin) if xmax > xmin else 0.5
pady = 0.2 * (ymax - ymin) if ymax > ymin else 0.5

# --- Initialize plot with first frame ---
d0 = load(files[0])
pa0 = d0["arrays"]["fluid"]
H0 = pa0.height if hasattr(pa0, "height") else pa0.rho
t0 = float(d0["solver_data"]["t"])

plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))
sc = ax.scatter(pa0.x, pa0.y, s=30, c=H0, marker='s', cmap='viridis')  # s=size in points²
plt.colorbar(sc, ax=ax, label="Height (rho)")
ax.set_aspect("equal")
ax.set_xlim(xmin - padx, xmax + padx)
ax.set_ylim(ymin - pady, ymax + pady)
sc.set_clim(vmin, vmax)  
ttl = ax.set_title(f"t = {t0:.4f}")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.tight_layout()
plt.show(block=False)

# --- Animate: loop through frames in time order ---
print("\nPlaying animation (Ctrl+C to stop)...")
try:
    for i in cycle(range(len(files))):
        d = load(files[i])
        pa = d["arrays"]["fluid"]
        t = float(d["solver_data"]["t"])
        H = pa.height if hasattr(pa, "height") else pa.rho
        
        # Update scatter plot data
        sc.set_offsets(np.c_[pa.x, pa.y])
        sc.set_array(H)
        ttl.set_text(f"t = {t:.4f}  (frame {i+1}/{len(files)})")
        plt.pause(0.1)  # Animation speed
except KeyboardInterrupt:
    print("\nAnimation stopped.")
