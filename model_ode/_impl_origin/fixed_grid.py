from .solvers import FixedGridODESolver
from .rk_common import rk4_alt_step_func
from .misc import Perturb
import torch
import copy

class Euler(FixedGridODESolver):
    order = 1

    def _step_func(self, func, t0, dt, t1, y0, \
                   sol, t_end, t_data, t_K, tau):
        
        integro = self.integration1(sol, t_end, t_data, t_K)
        
        ######
        f0 = func(t0, y0, integro, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
        ######
        return dt * f0, f0


class Midpoint(FixedGridODESolver):
    order = 2

    def _step_func(self, func, t0, dt, t1, y0, \
                   sol, t_end, t_data, t_K, tau):
        
        integro = self.integration1(sol, t_end, t_data, t_K)
        
        #######        
        half_dt = 0.5 * dt
        f0 = self.func(t0, y0, integro, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
        y_mid = y0 + f0 * half_dt
        #######
        
        t_data_half = copy.deepcopy(t_data)/tau
        t_data_half[-1] = t1-half_dt
        t_data[-1] = t1-half_dt
        sol_half = torch.cat((sol[:-1],y_mid.reshape([1, *y_mid.shape])),dim=0)
        integro_h = self.integration1(sol_half, t_end=(t1-half_dt)*tau, t_data=t_data_half*tau, t_K=t_K)
        
        return dt * func(t0 + half_dt, y_mid, integro_h), f0


class RK4(FixedGridODESolver):
    order = 4

    def _step_func(self, func, t0, dt, t1, y0):
        f0 = func(t0, y0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
        return rk4_alt_step_func(func, t0, dt, t1, y0, f0=f0, perturb=self.perturb), f0
