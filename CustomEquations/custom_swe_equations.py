from pysph.sph.equation import Equation

class SWEPressureFromRho(Equation):
    """
    a <- -g * sum_j Vj * ∇W_ij   with  Vj = m_j / rho_j  and rho ≡ H.
    """
    def __init__(self, dest, sources, g=9.81):
        super().__init__(dest, sources)
        self.g = g
    """
    d_idx -> destination index
    d_au -> acceleration (x) array
    d_av -> acceleration (y) array
    """
    def initialize(self, d_idx, d_au, d_av):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
    """
    s_idx -> source index (particle arrays that we consider for calculations)
    s_m -> mass array of source particles
    s_rho -> rho array of source particles
    DWIJ -> Del(W) (kernel gradient) between particle i and j
    """
    def loop(self, d_idx, s_idx, d_au, d_av, s_m, s_rho, DWIJ):
        Vj = s_m[s_idx] / (s_rho[s_idx] + 1e-12)
        Hij = s_rho[s_idx]
        d_au[d_idx] += -self.g * Vj * Hij * DWIJ[0]
        d_av[d_idx] += -self.g * Vj * Hij * DWIJ[1]

class SWEViscosity(Equation):
    """
    a <- nu * sum_j Vj * (u_j - u_i)/h_j * ∇W_ij  (component-wise).
    """
    def __init__(self, dest, sources, nu):
        super().__init__(dest, sources)
        self.nu = nu
    def loop(self, d_idx, s_idx, d_au, d_av, s_m, s_rho, s_h,
             d_u, d_v, s_u, s_v, DWIJ):
        Vj = s_m[s_idx] / (s_rho[s_idx] + 1e-12)
        fac = self.nu * Vj / (s_h[s_idx] + 1e-12)
        d_au[d_idx] += fac * (s_u[s_idx] - d_u[d_idx]) * DWIJ[0]
        d_av[d_idx] += fac * (s_v[s_idx] - d_v[d_idx]) * DWIJ[1]