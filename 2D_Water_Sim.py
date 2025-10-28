#Supporting
import os
from numpy import mgrid, sqrt, ones_like
import numpy as np
from pysph.base.utils import get_particle_array
#Main application
from pysph.solver.application import Application

#Create equations
from pysph.sph.equation import Group
from pysph.sph.basic_equations import SummationDensity, XSPHCorrection
from CustomEquations.custom_swe_equations import SWEPressureFromRho, SWEViscosity

#Solvers/Integrators
from pysph.solver.solver import Solver
from pysph.sph.integrator import EPECIntegrator
from pysph.sph.integrator_step import WCSPHStep

from pysph.base.kernels import CubicSpline

"""
TODOS:
* Create custom equation to calculate the momentum (Forces acting on a particle being)
    * F_pressure_i = -g del(h_i) = -g*summ(V_j*del(kernel))
    * F_viscous_i = dynamic_visc * summ(V_j*(u_j - u_i) / h_j)*grad(kernel)
    * g -> gravity, V_j is volume (constant), h is height (/rho)

* 
"""

#Subclassing from Application
class swe_sim(Application):
    def initialize(self):
        self.dx = 0.02
        self.hdx = 1.3
        self.ro = 1.0

    def create_particles(self):
        dx = self.dx
        hdx = self.hdx
        ro = self.ro
        name = 'fluid'

        x, y = mgrid[-1.05:1.05+1e-4:dx, -1.05:1.05+1e-4:dx]
        x = x.ravel() #flatten arrays
        y = y.ravel()

        m = ones_like(x)*dx*dx*ro #particle mass array
        h = ones_like(x)*hdx*dx #smoothing radius of particles
        rho = ones_like(x) * ro #initial density of particles
        u = -100*x
        v = 100*y

        # remove particles outside the circle
        indices = []
        for i in range(len(x)):
            if sqrt(x[i]*x[i] + y[i]*y[i]) - 1 > 1e-10:
                indices.append(i)
        pa = get_particle_array(x=x, y=y, m=m, rho=rho, h=h, u=u, v=v,
                                name=name)
        pa.remove_particles(indices)

        #Setup properties
        # add required "previous state" properties
        for prop in ('rho0','u0','v0','w0','x0','y0','z0', 'arho', 'ax', 'ay', 'az'):
            pa.add_property(prop)


        # initialize them to current values
        pa.rho0[:] = pa.rho
        pa.u0[:]   = pa.u
        pa.v0[:]   = pa.v
        pa.w0[:]   = 0.0            # 2D
        pa.x0[:]   = pa.x
        pa.y0[:]   = pa.y
        pa.z0[:]   = 0.0            # 2D
        # initialize to zeros
        pa.arho[:] = 0.0
        pa.ax[:] = 0.0; pa.ay[:] = 0.0; pa.az[:] = 0.0

        print(f"Number of particles :: {pa.get_number_of_particles()}")

        return [pa]
        
    def create_scheme(self):
        return None #Create custom scheme
    
    def create_solver(self):
        kernel = CubicSpline(dim=2) #2D particles
        
        integrator = EPECIntegrator(fluid=WCSPHStep())
        dt = 0.25 * self.hdx * self.dx / 100.0
        tf = 0.5
        return Solver(kernel=kernel, dim=2, integrator=integrator, dt=dt, tf=tf)

    def create_equations(self):
        return [
            # Updating rho (pseudo height)
            Group([
                SummationDensity(dest='fluid', sources=['fluid']),
            ]),
            # Updating accelerations
            Group([
                SWEPressureFromRho(dest='fluid', sources=['fluid'], g=9.81),
                SWEViscosity(dest='fluid', sources=['fluid'], nu=1e-3),  # tune ν
            ])
        ]
    
    def _compute_results(self):
        from pysph.solver.utils import iter_output
        from collections import defaultdict
        data = defaultdict(list)

        for sd, array in iter_output(self.output_files, 'fluid'):
            t = sd['t']
            m, u, v, rho, x, y = array.get('m', 'u', 'v', 'rho', 'x', 'y')
            ke = 0.5 * np.sum(m * (u*u + v*v)) #KE

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
            data['H-max'].append(rho.max())
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
    
