# Shallow Water SPH Simulation

2D shallow water simulation using **Smoothed Particle Hydrodynamics (SPH)**, built on [PySPH](https://github.com/pypr/pysph). Particles represent a fluid surface; `rho` is used as pseudo-height `H`. Output is written to disk and visualized as an animated scatter plot colored by height.

## References

This implementation follows ideas from:

- Solenthaler et al. — [SPH Shallow Water](https://matthias-research.github.io/pages/publications/SPHShallow.pdf)
- Han & Lee — [SPH Based Shallow Water Simulation](https://link.springer.com/article/10.1007/s00371-010-0439-9)

Local copies of related papers are included in the repo root.

## Features

- Custom SWE pressure and viscosity equations with **Spiky** and **Viscosity** kernels (computed manually inside the equations)
- **SummationDensity** (CubicSpline kernel via PySPH) for height estimation
- Multiple initial fluid patches via circular masks
- Square domain with boundary particles, repulsion forces, and hard velocity reflection at walls
- Post-processing summary saved to `results.npz` (kinetic energy, height range, mass, geometry)

## Project structure

```
SPH_Water_Sim/
├── 2D_Water_Sim.py              # Main simulation (PySPH Application)
├── view_npz_folder_2d.py        # Frame-by-frame visualization
├── CustomEquations/
│   └── custom_swe_equations.py  # SWEStep, pressure, viscosity, boundary forces
├── requirements.txt             # Python dependencies
└── 2D_Water_Sim_output/         # Generated output (gitignored)
```

## Requirements

- Python 3.11+ (tested with PySPH 1.0b2)
- See `requirements.txt`:

| Package     | Purpose                          |
|-------------|----------------------------------|
| PySPH       | SPH solver, integrator, NNPS     |
| numpy       | Arrays, initial particle layout  |
| matplotlib  | Visualization (`view_npz_*`)     |

PySPH pulls in additional packages automatically (Cython, compyle, cyarray, etc.).

## Setup

From the project root:

```powershell
python -m venv pysph_env
.\pysph_env\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you already have a local `pysph_env` folder, just activate it:

```powershell
.\pysph_env\Scripts\Activate.ps1
```

## Usage

### Run simulation + visualize

Clear old output first so frames are not mixed across runs:

```powershell
Remove-Item -Path "2D_Water_Sim_output" -Recurse -Force
python 2D_Water_Sim.py
python view_npz_folder_2d.py
```

One-liner:

```powershell
Remove-Item -Path "2D_Water_Sim_output" -Recurse -Force; python 2D_Water_Sim.py; python view_npz_folder_2d.py
```

### Visualize an existing run only

```powershell
python view_npz_folder_2d.py
python view_npz_folder_2d.py path\to\other_output_folder
```

Press **Ctrl+C** to stop the animation loop.

## Output

After a run, PySPH writes timestep files under `2D_Water_Sim_output/`. Post-processing also creates:

- `2D_Water_Sim_output/results.npz` — time series of kinetic energy, min/max height, mass, and particle bounds

The viewer reads PySPH output directly (not `results.npz`) and colors fluid particles by height (`rho`).

## Configuration

Main parameters live in `2D_Water_Sim.py` inside `swe_sim.initialize()` and related methods:

| Parameter        | Location              | Notes                                      |
|------------------|-----------------------|--------------------------------------------|
| `dx`             | `initialize()`        | Particle spacing; smaller = more particles |
| `hdx`            | `initialize()`        | Smoothing length ratio (`h = hdx * dx`)    |
| `domain_size`    | `initialize()`        | Half-width of square domain                |
| `list_of_centers`| `create_particles()`  | Fluid patches: `(x, y, radius)` tuples     |
| `g`, `nu`        | `create_equations()`  | Gravity and viscosity                      |
| `tf`, `pfreq`    | `create_solver()`     | Final time and output frequency            |

Example — two circular fluid blobs:

```python
list_of_centers = [(1, 1, 1), (-1, -1, 1)]
```

Custom physics equations are in `CustomEquations/custom_swe_equations.py`.

## Simulation pipeline

Each timestep:

1. **SummationDensity** — compute height `H` (`rho`) from particle layout
2. **SWEPressureFromRho** — pressure gradient (Spiky kernel)
3. **SWEViscosity** — viscous diffusion (Viscosity kernel)
4. **BoundaryForce** — cancel acceleration into walls near boundary particles
5. **SWEStep** — integrate positions/velocities with hard reflection at domain edges

## Troubleshooting

| Issue | Likely cause | Fix |
|-------|--------------|-----|
| `LinkedListNNPS requires too many cells` | Particles escaped the domain | Reduce `dt`, strengthen boundaries, or lower initial velocity |
| Mixed / out-of-order frames in viewer | Old output not cleared | Delete `2D_Water_Sim_output` before re-running |
| Only one fluid patch appears | Bounding box typo in `create_circular_patches` | Ensure `ymax` uses `c[1] + c[2]`, not `-` |
| PySPH compile errors on first run | Cython building equation kernels | Wait for first-run compile; ensure Cython is installed |

