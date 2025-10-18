import os

from numpy import mgrid, sqrt, ones_like
import numpy as np

from pysph.base.utils import get_particle_array

from pysph.solver.application import Application

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

        x, y = mgrid()
        return
    def create_scheme(self):
        return
    


if __name__ == '__main__':
    app = swe_sim()
    app.run()
    