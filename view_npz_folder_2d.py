import sys, os, numpy as np, matplotlib.pyplot as plt
from itertools import cycle
from pysph.solver.utils import get_files, load

outdir = sys.argv[1] if len(sys.argv) > 1 else "2D_Water_Sim_output"
files = get_files(outdir)

if not files:
    raise SystemExit(f"No output files in {outdir}")

x0,y0 = load(files[0])['arrays']['fluid'].get('x','y', only_real_particles=True)

for i,f in enumerate(files[1:], 2):
    pa = load(f)['arrays']['fluid']
    x,y = pa.get('x','y', only_real_particles=True)
    disp = np.hypot(x - x0, y - y0)
    print(i, "max|Δx,y|=", float(disp.max()), " mean|Δ|=", float(disp.mean()))
    x0, y0 = x, y

times = []
for fp in files:
    d = load(fp)
    t = float(d["solver_data"]["t"])
    times.append((t, fp))
times.sort(key=lambda x: x[0])
files = [fp for _, fp in times]

# --- find max bounds
xmin = ymin = np.inf
xmax = ymax = -np.inf

vmin = np.inf
vmax = -np.inf

for fpath in files:
    dic = load(fpath)
    particle_arr = dic["arrays"]["fluid"]
    x, y = particle_arr.x, particle_arr.y
    H = particle_arr.height if hasattr(particle_arr, "height") else particle_arr.rho

    xmin = min(xmin, float(x.min()))
    xmax = max(xmax, float(x.max()))

    ymin = min(ymin, float(y.min()))
    ymax = max(ymax, float(y.max()))
    vmin = min(vmin, float(H.min()))
    vmax = max(vmax, float(H.max()))

padx = 0.2*(xmax - xmin or 1.0)
pady = 0.2*(ymax - ymin or 1.0)

# Init Plot
d0 = load(files[0]); 
pa0 = d0["arrays"]["fluid"]
float(pa0.u.max()), float(pa0.v.max())
H0 = pa0.height if hasattr(pa0, "height") else pa0.rho


# Looking at the data:


plt.ion()
fig, ax = plt.subplots()
sc = ax.scatter(pa0.x, pa0.y, s=4, c=H0)
plt.colorbar(sc, ax=ax, label=("height" if hasattr(pa0, "height") else "rho"))
ax.set_aspect("equal")
ax.set_xlim(xmin - padx, xmax + padx)
ax.set_ylim(ymin - pady, ymax + pady)   # prevents y from looking cut off
sc.set_clim(vmin, vmax)
ttl = ax.set_title(os.path.basename(files[0]))
plt.tight_layout()
plt.show(block=False)

# --- loop forever in correct time order ---
for i in cycle(range(len(files))):
    d = load(files[i]); 
    pa = d["arrays"]["fluid"] #particle array

    H = pa.height if hasattr(pa, "height") else pa.rho
    sc.set_offsets(np.c_[pa.x, pa.y])
    sc.set_array(H)
    ttl.set_text(os.path.basename(files[i]))
    plt.pause(0.3)