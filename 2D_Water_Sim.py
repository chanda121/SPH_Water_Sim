import os
from numpy import mgrid, sqrt, ones_like
import numpy as np
from pysph.base.utils import get_particle_array
from pysph.solver.application import Application
from pysph.sph.equation import Group
from pysph.sph.basic_equations import SummationDensity
from CustomEquations.custom_swe_equations import (
    SWEPressureFromRho, SWEViscosity, SWEStep, BoundaryForce, BoundaryStep
)
from pysph.solver.solver import Solver
from pysph.sph.integrator import EPECIntegrator
from pysph.base.kernels import CubicSpline


class swe_sim(Application):
    def initialize(self):
        # Particle spacing - larger = fewer particles, larger physical scale
        self.dx = 0.05
        self.hdx = 2.0     # Smoothing length ratio (standard SPH value)
        self.ro = 1.0       # Reference density (= reference height)
        self.radius = 1.0   # Radius of initial fluid circle
        
        # Domain bounds for boundary particles
        self.domain_size = 2.0  # Half-width of domain 

    def create_particles(self):
        dx = self.dx
        hdx = self.hdx
        ro = self.ro
        R = self.radius
        
        # Create initial fluid particles
        list_of_centers = [(1,1,1), (-1,-1,1)]
        x, y = self.create_circular_patches(dx, list_of_centers)
        
        # Particle properties
        m = ones_like(x) * dx * dx * ro   # mass = area * density
        h = ones_like(x) * hdx * dx       # smoothing length
        rho = ones_like(x) * ro           # initial density/height
        
        # Initial velocity - start at rest
        u = np.zeros_like(x)
        v = np.zeros_like(x)
 
        # Create fluid particle array
        fluid = get_particle_array(
            x=x, y=y, m=m, rho=rho, h=h, u=u, v=v,
            name='fluid'
        )

        # Add required properties for the integrator
        for prop in ('rho0', 'u0', 'v0', 'x0', 'y0', 'au', 'av'):
            fluid.add_property(prop)

        # Initialize fluid properties
        fluid.rho0[:] = fluid.rho
        fluid.u0[:] = fluid.u
        fluid.v0[:] = fluid.v
        fluid.x0[:] = fluid.x
        fluid.y0[:] = fluid.y
        fluid.au[:] = 0.0
        fluid.av[:] = 0.0

        # Create boundary particles with normals
        bx, by, bnx, bny = self.create_boundary_particles(dx, self.domain_size)
        
        boundary = get_particle_array(
            x=bx, y=by,
            m=ones_like(bx) * dx * dx * ro,
            rho=ones_like(bx) * ro,
            h=ones_like(bx) * hdx * dx,
            u=np.zeros_like(bx),
            v=np.zeros_like(bx),
            name='boundary'
        )
        
        # Add normals to boundary particles
        boundary.add_property('nx')
        boundary.add_property('ny')
        boundary.nx[:] = bnx
        boundary.ny[:] = bny

        print(f"Number of fluid particles: {fluid.get_number_of_particles()}")
        print(f"Number of boundary particles: {boundary.get_number_of_particles()}")
        print(f"Particle spacing dx = {dx}")
        print(f"Smoothing length h = {hdx * dx:.4f}")
        print(f"Domain size: [{-self.domain_size}, {self.domain_size}]")

        return [fluid, boundary]
   
    def create_circular_patches(self, dx, list_of_centers):
        """
        dx - particle distance
        list_of_centers = [(x1, y1, r1), ...]
        """
        xmin = min(c[0] - c[2] for c in list_of_centers) - dx 
        xmax = max(c[0] + c[2] for c in list_of_centers) + dx
        ymin = min(c[1] - c[2] for c in list_of_centers) - dx
        ymax = max(c[1] + c[2] for c in list_of_centers) + dx

        #Get initial bounding box
        x, y = mgrid[xmin:xmax:dx, ymin:ymax:dx]
        x, y = x.ravel(), y.ravel()

        #Create masks
        mask = np.zeros(len(x), dtype=bool)
        for cx, cy, r in list_of_centers:
            mask |= ((x-cx)**2 + (y-cy)**2) <= r**2 #do an OR to add all values inside circles to mask as true
            
        return x[mask], y[mask]

    def create_boundary_particles(self, dx, half_width):
        """
        Create boundary particles around a square domain with normals.
        Returns x, y, nx, ny arrays.
        
        Normals point INTO the fluid (away from wall).
        """
        n_layers = 3
        edge = np.arange(-half_width, half_width + dx, dx)
        inner_edge = edge[1:-1]  # exclude corners
        
        x_list = []
        y_list = []
        nx_list = []
        ny_list = []
        
        for layer in range(n_layers):
            offset = layer * dx
            
            # Bottom wall: normal points up (0, 1)
            x_list.append(edge)
            y_list.append(np.full_like(edge, -half_width - offset))
            nx_list.append(np.zeros_like(edge))
            ny_list.append(np.ones_like(edge))
            
            # Top wall: normal points down (0, -1)
            x_list.append(edge)
            y_list.append(np.full_like(edge, half_width + offset))
            nx_list.append(np.zeros_like(edge))
            ny_list.append(np.full_like(edge, -1.0))
            
            # Left wall: normal points right (1, 0)
            x_list.append(np.full_like(inner_edge, -half_width - offset))
            y_list.append(inner_edge)
            nx_list.append(np.ones_like(inner_edge))
            ny_list.append(np.zeros_like(inner_edge))
            
            # Right wall: normal points left (-1, 0)
            x_list.append(np.full_like(inner_edge, half_width + offset))
            y_list.append(inner_edge)
            nx_list.append(np.full_like(inner_edge, -1.0))
            ny_list.append(np.zeros_like(inner_edge))
        
        x = np.concatenate(x_list)
        y = np.concatenate(y_list)
        nx = np.concatenate(nx_list)
        ny = np.concatenate(ny_list)
        
        return x, y, nx, ny

    def create_scheme(self):
        return None

    def create_solver(self):
        kernel = CubicSpline(dim=2)
        
        # Integrator for fluid and static boundary
        # Pass domain bounds to integrator for hard clamping
        ds = self.domain_size
        integrator = EPECIntegrator(
            fluid=SWEStep(xmin=-ds, xmax=ds, ymin=-ds, ymax=ds, margin=self.dx),
            boundary=BoundaryStep()
        )
        
        # Time stepping
        # CFL: dt < h / c, where c = sqrt(g*H) ≈ 3 for g=10, H=1
        c0 = 5.0  # characteristic wave speed
        dt = 0.2 * self.hdx * self.dx / c0
        tf = 5.0   
        pfreq = 15  # output every 15 steps
        
        print(f"Time step dt = {dt:.6f}")
        print(f"Final time tf = {tf}")
        
        return Solver(
            kernel=kernel, dim=2, integrator=integrator,
            dt=dt, tf=tf, pfreq=pfreq
        )

    def create_equations(self):
        # Boundary influence distance - should be large enough to catch fast particles
        # Using smoothing length scale (hdx * dx) is a good choice
        b = self.hdx * self.dx * 2.0  # 2x smoothing length
        
        return [
            # Step 1: Compute height (rho) from particle distribution
            Group([
                SummationDensity(dest='fluid', sources=['fluid']),
            ]),
            # Step 2: Compute accelerations from pressure and viscosity
            Group([
                SWEPressureFromRho(dest='fluid', sources=['fluid'], g=9.8, dx=self.dx),
                SWEViscosity(dest='fluid', sources=['fluid'], nu=0.1, dx=self.dx),
            ]),
            # Step 3: Boundary repulsion force
            # - repulsion_coeff: strength of distance-based push-away
            # - damping: velocity damping for particles moving toward wall
            # - c: force reflection intensity
            Group([
                BoundaryForce(dest='fluid', sources=['boundary'], b=b, c=0.5),
            ])
        ]

    def _compute_results(self):
        """Compute and save simulation results to .npz file."""
        from pysph.solver.utils import iter_output
        from collections import defaultdict
        data = defaultdict(list)

        for sd, array in iter_output(self.output_files, 'fluid'):
            t = sd['t']
            m, u, v, rho, x, y = array.get('m', 'u', 'v', 'rho', 'x', 'y')
            ke = 0.5 * np.sum(m * (u*u + v*v))  # Kinetic energy

            # Geometry
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            mx = m.sum()
            xc = (m * x).sum() / mx
            yc = (m * y).sum() / mx
            r_max = np.sqrt((x - xc)**2 + (y - yc)**2).max()

            data['t'].append(t)
            data['ke'].append(ke)
            data['H_min'].append(rho.min())
            data['H_max'].append(rho.max())
            data['mass'].append(m.sum())
            data['x_min'].append(x_min); data['x_max'].append(x_max)
            data['y_min'].append(y_min); data['y_max'].append(y_max)
            data['xc'].append(xc); data['yc'].append(yc)
            data['r_max'].append(r_max)

        for k in list(data.keys()):
            data[k] = np.asarray(data[k])
        np.savez(os.path.join(self.output_dir, 'results.npz'), **data)

    def post_process(self, info_file_or_dir):
        if self.rank > 0:
            return
        self.read_info(info_file_or_dir)
        if not self.output_files:
            return
        self._compute_results()


if __name__ == '__main__':
    app = swe_sim()
    app.run()
    app.post_process(app.info_filename)
