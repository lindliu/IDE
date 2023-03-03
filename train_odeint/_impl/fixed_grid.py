from .solvers import FixedGridODESolver
from .rk_common import rk4_alt_step_func
from .misc import Perturb


class Euler(FixedGridODESolver):
    order = 1

    def _step_func(self, func, t0, dt, t1, y0, solution, time_grid, j):
        # f0 = func(t0, y0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
        # func.mm = self._memory(func, solution, time_grid)
        # print(func.mm.shape)
        
        # print(t0.item()==0)
        f0 = func(t0, y0, solution, time_grid, j)
        # print('asdfsfd', solution)
        # print(memory.sum())
        return dt * f0, f0

    # def _memory(self, func, solution, time_grid):
    #     import torch
    #     S, I, R = torch.split(solution,1,dim=2)
        
    #     dt = 0.4103
    #     batchsize = S.shape[1]
    #     print(batchsize)
        
    #     t = torch.flip(time_grid, [0]).reshape([-1,1])
    #     # print(t)
    #     gamma = func.memory(t)
    #     gamma = gamma.repeat(1,batchsize)
    #     # print(I[:,:,0].shape)
    #     # return func.memory(torch.tensor([[1.]]).cuda())
    #     integro = I[:,:,0] * gamma * dt
    #     # print(integro)
    #     return integro.sum(0).reshape(batchsize,1)
    
class Midpoint(FixedGridODESolver):
    order = 2

    def _step_func(self, func, t0, dt, t1, y0):
        half_dt = 0.5 * dt
        f0 = func(t0, y0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
        y_mid = y0 + f0 * half_dt
        return dt * func(t0 + half_dt, y_mid), f0


class RK4(FixedGridODESolver):
    order = 4

    def _step_func(self, func, t0, dt, t1, y0):
        f0 = func(t0, y0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
        return rk4_alt_step_func(func, t0, dt, t1, y0, f0=f0, perturb=self.perturb), f0
