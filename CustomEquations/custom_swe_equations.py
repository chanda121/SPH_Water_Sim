"""
Custom SPH equations for Shallow Water Simulation.
Based on Han & Lee "SPH Based Shallow Water Simulation"
"""
from pysph.sph.equation import Equation
from pysph.sph.integrator_step import IntegratorStep
from numpy import pi
class SWEStep(IntegratorStep):
    """
    EPEC integrator for shallow water with hard boundary reflection.
    
    Integrates position (x,y) and velocity (u,v).
    Height (rho) is computed via SummationDensity each step.
    Includes hard position clamp + velocity reflection at boundaries.
    """
    def __init__(self, xmin=-5.0, xmax=5.0, ymin=-5.0, ymax=5.0, margin=0.02, restitution=1.0):
        """
        margin: Distance from wall to clamp particles (should be > 0)
        restitution: Coefficient of restitution (1.0 = perfect elastic, 0.5 = lose half velocity)
        """
        self.xmin = xmin + margin
        self.xmax = xmax - margin
        self.ymin = ymin + margin
        self.ymax = ymax - margin
        self.restitution = restitution
    
    def initialize(self, d_idx, d_x, d_y, d_u, d_v,
                   d_x0, d_y0, d_u0, d_v0):
        # Store state at start of timestep
        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]

    def stage1(self, d_idx, d_x, d_y, d_u, d_v,
               d_au, d_av, d_x0, d_y0, d_u0, d_v0, dt):
        # Half-step predictor
        dtb2 = 0.5 * dt
        d_u[d_idx] = d_u0[d_idx] + dtb2 * d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dtb2 * d_av[d_idx]
        d_x[d_idx] = d_x0[d_idx] + dtb2 * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dtb2 * d_v[d_idx]
        
        # Hard reflect at boundaries
        if d_x[d_idx] < self.xmin:
            d_x[d_idx] = self.xmin
            d_u[d_idx] = -self.restitution * d_u[d_idx]
        elif d_x[d_idx] > self.xmax:
            d_x[d_idx] = self.xmax
            d_u[d_idx] = -self.restitution * d_u[d_idx]
        
        if d_y[d_idx] < self.ymin:
            d_y[d_idx] = self.ymin
            d_v[d_idx] = -self.restitution * d_v[d_idx]
        elif d_y[d_idx] > self.ymax:
            d_y[d_idx] = self.ymax
            d_v[d_idx] = -self.restitution * d_v[d_idx]

    def stage2(self, d_idx, d_x, d_y, d_u, d_v,
               d_au, d_av, d_x0, d_y0, d_u0, d_v0, dt):
        # Full-step corrector
        d_u[d_idx] = d_u0[d_idx] + dt * d_au[d_idx]
        d_v[d_idx] = d_v0[d_idx] + dt * d_av[d_idx]
        d_x[d_idx] = d_x0[d_idx] + dt * d_u[d_idx]
        d_y[d_idx] = d_y0[d_idx] + dt * d_v[d_idx]
        
        # Hard reflect at boundaries
        if d_x[d_idx] < self.xmin:
            d_x[d_idx] = self.xmin
            d_u[d_idx] = -self.restitution * d_u[d_idx]
        elif d_x[d_idx] > self.xmax:
            d_x[d_idx] = self.xmax
            d_u[d_idx] = -self.restitution * d_u[d_idx]
        
        if d_y[d_idx] < self.ymin:
            d_y[d_idx] = self.ymin
            d_v[d_idx] = -self.restitution * d_v[d_idx]
        elif d_y[d_idx] > self.ymax:
            d_y[d_idx] = self.ymax
            d_v[d_idx] = -self.restitution * d_v[d_idx]


class SWEPressureFromRho(Equation):
    """
    Pressure gradient for shallow water using SPIKY kernel gradient.
    
    Spiky kernel gradient (2D):
    del(W_spiky) = -30/(PI*l^5) * (l-r)^2 * unit_r    for 0 <= r <= l
    
    Pressure force: a_i = -g * summ( V_j * del(W_ij) )
    """
    def __init__(self, dest, sources, g=9.81, dx=0.05):
        super().__init__(dest, sources)
        self.g = g
        self.V = dx * dx  # Constant particle volume (2D area)

    def initialize(self, d_idx, d_au, d_av):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_au, d_av, d_x, d_y, s_x, s_y, s_h, d_rho, s_rho):
        # Compute distance vector (from j to i)
        xij = d_x[d_idx] - s_x[s_idx]
        yij = d_y[d_idx] - s_y[s_idx]
        r2 = xij*xij + yij*yij
        r = (r2 + 1e-12) ** 0.5
        
        # Smoothing length
        l = s_h[s_idx]
        
        # Spiky gradient (only if 0 < r < l)
        if r < l and r > 1e-12:
            # Del(W_spiky) = -30/(PI*l^5) * (l-r)^2 * (r_vec/r)
            l_5 = l * l * l * l * l
            coeff = -30.0 / (pi * l_5)
            grad_mag = coeff * (l - r) * (l - r) / r
            
            grad_Wx = grad_mag * xij
            grad_Wy = grad_mag * yij
            
            # Height (rho) at particles - use symmetric average for stability
            h_i = d_rho[d_idx]
            h_j = s_rho[s_idx]
            h_avg = 0.5 * (h_i + h_j)
            
            # Pressure force: -g * V * h * del(W)
            # Higher water -> more pressure -> particles spread out
            fac = -self.g * self.V * h_avg
            d_au[d_idx] += fac * grad_Wx
            d_av[d_idx] += fac * grad_Wy


class SWEViscosity(Equation):
    """
    Viscous diffusion for shallow water using VISCOSITY kernel Laplacian.
    
    Viscosity kernel Laplacian (2D):
    Laplacian(W_visc) = 40/(PI*l^5) * (l - r)    for 0 <= r <= l
    
    Viscous force: a_i = nu * sum( V_j * (u_j - u_i) / h_j * laplacian(W_ij) )
    """
    def __init__(self, dest, sources, nu, dx=0.05):
        super().__init__(dest, sources)
        self.nu = nu
        self.V = dx * dx

    def loop(self, d_idx, s_idx, d_au, d_av,
             d_u, d_v, s_u, s_v, s_rho, d_x, d_y, s_x, s_y, s_h):
        # Compute distance
        xij = d_x[d_idx] - s_x[s_idx]
        yij = d_y[d_idx] - s_y[s_idx]
        r2 = xij*xij + yij*yij
        r = (r2 + 1e-12) ** 0.5
        
        # Smoothing length and height
        l = s_h[s_idx]
        hj = s_rho[s_idx]
        if hj < 0.01:
            hj = 0.01
        
        # Viscosity Laplacian (only if r < l)
        if r < l:
            # laplacian(W_visc) = 20/(PI*l^5) * (l - r)
            l_5 = l * l * l * l * l
            laplacian_W = 40.0 / (pi * l_5) * (l - r)
            
            # Viscous acceleration: nu * V * (u_j - u_i) / h_j * laplacian(W_ij)
            fac = self.nu * self.V * laplacian_W / hj
            d_au[d_idx] += fac * (s_u[s_idx] - d_u[d_idx])
            d_av[d_idx] += fac * (s_v[s_idx] - d_v[d_idx])


# ============ BOUNDARY CLASSES ============

class BoundaryStep(IntegratorStep):
    """Dummy integrator for static boundary particles."""
    def initialize(self):
        pass
    def stage1(self):
        pass
    def stage2(self):
        pass


class BoundaryForce(Equation):
    """
    Boundary force from paper Eq. 17-18.
    
    f_boundary = alpha * (f · n) * n   [Eq. 17]
    alpha = c * (1 - l / b)            [Eq. 18]
    
    When particle is near boundary and has acceleration toward it (f·n < 0),
    we subtract that inward component to prevent acceleration into the wall.
    
    Boundary particles must have nx, ny (normal pointing INTO fluid).
    """
    def __init__(self, dest, sources, b=0.1, c=1.0):
        super().__init__(dest, sources)
        self.b = b  # Influence distance
        self.c = c  # Intensity (1.0 = fully cancel inward acceleration)
    
    def loop(self, d_idx, s_idx, d_au, d_av, d_x, d_y, s_x, s_y, s_nx, s_ny):
        # Distance from fluid particle to boundary particle
        dx = d_x[d_idx] - s_x[s_idx]
        dy = d_y[d_idx] - s_y[s_idx]
        r = (dx*dx + dy*dy + 1e-12) ** 0.5
        
        # Only apply within influence distance
        if r < self.b:
            # Normal (points INTO fluid)
            nx = s_nx[s_idx]
            ny = s_ny[s_idx]
            
            # alpha = c * (1 - l/b)  [Eq. 18]
            # Stronger effect closer to boundary
            alpha = self.c * (1.0 - r / self.b)
            
            # Project acceleration onto normal: f · n
            f_dot_n = d_au[d_idx] * nx + d_av[d_idx] * ny
            
            # If acceleration points toward boundary (f·n < 0), cancel it
            if f_dot_n < 0.0:
                # Subtract the inward component: a -= alpha * (f·n) * n
                d_au[d_idx] -= alpha * f_dot_n * nx
                d_av[d_idx] -= alpha * f_dot_n * ny