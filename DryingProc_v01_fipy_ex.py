# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 00:57:47 2025

@author: SebyG
"""
'''

# %%
import fipy as fp
import numpy as np

# %%

# Example domain size (Lx x Ly)
length_x = 0.5  # meters, direction of film travel
length_y = 0.2  # meters, cross-stream height

nx = 50
ny = 20

mesh = fp.Grid2D(dx=length_x/nx, dy=length_y/ny, nx=nx, ny=ny)

c_g = fp.CellVariable(name="solvent_concentration_gas", 
                      mesh=mesh, 
                      value=0.0)  # initial guess


# %%

D_g = 1e-5  # diffusion coeff of solvent in gas, m^2/s
# For convection, define a velocity field. 
# Example: uniform cross flow from right to left
U_gas = -0.1  # m/s  (negative means flow from right -> left in x-direction)
V_gas = 0.0   # no vertical component for simplicity
velocity_field = (U_gas, V_gas)


# %%

# Identify right boundary
x_right = mesh.x == length_x  # boolean array

# Suppose we set c_g = 0 at the injection segments
c_g.constrain(0.0, where=x_right)

# %%
x_left = mesh.x == 0
#c_g.faceGrad.dot(mesh.faceNormals)[x_left] = 0  # or use FiPy's built-in approach
c_g.faceGrad.constrain(0., where=mesh.facesLeft)



# %%
# On the lower boundary (y=0):
y_bottom = mesh.y == 0

# We can set a flux term or a source in the gas, e.g.:
# R_interface = k * (c_film - c_g)
# but we need a "virtual" c_film or a known value if the film concentration is given.

film_concentration = 1.0  # example constant or function of x, time
k_mass_transfer = 1e-3

# FiPy approach: add a source/sink term near that boundary
# The easiest way might be to add a term in the PDE for cells near y=0

# %%

eq = (fp.TransientTerm(var=c_g)
      + fp.ConvectionTerm(coeff=velocity_field, var=c_g)
      == fp.DiffusionTerm(coeff=D_g, var=c_g))

# If I have source_term then activate the following comment
#eq += source_term  # or eq == eq + ...

# %%
dt = 0.001  # time step
steps = 500

for step in range(steps):
    eq.solve(dt=dt)
    # Possibly log or save intermediate data


# %%

import matplotlib.pyplot as plt

fig = plt.figure()
viewer = fp.Viewer(vars=(c_g,))

for step in range(steps):
    eq.solve(dt=dt)
    if step % 50 == 0:
        viewer.plot()  # update plot

'''
"""
Modernized FiPy script for a 20x20 impingement problem
(similar to the classic Allen-Cahn example).

We use a semi-implicit approach so that FiPy does NOT complain about
"Terms with explicit Variables cannot mix with Terms with implicit Variables."
"""
"""
Pure diffusion in a 2D (20x20) domain.
The bottom boundary (y=0) is fixed at c=1.
All other boundaries are no-flux.
We solve transiently to watch the concentration diffuse upwards.
"""#!/usr/bin/env python

r"""
This example impinges three circular grains of nonconserved phase field

>>> from fipy import CellVariable, Grid2D, Viewer, TransientTerm, DiffusionTerm
>>> from fipy.tools import numerix

>>> dx = dy = 1.
>>> nx = ny = 20
>>> mesh = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)

>>> phase = CellVariable(
...     name='phase',
...     mesh=mesh,
...     value=0.)

>>> x, y = mesh.cellCenters
>>> radius = 5.

>>> phase.setValue(
...     1.,
...     where=(x - 10.)**2 + (y - 5.)**2 < radius**2)
>>> phase.setValue(
...     -1.,
...     where=(x - 5.)**2 + (y - 15.)**2 < radius**2)
>>> phase.setValue(
...     -1.,
...     where=(x - 15.)**2 + (y - 15.)**2 < radius**2)

We solve the Allen-Cahn equation

.. math::

   \frac{\partial \phi}{\partial t} = \nabla^2 \phi + \phi - \phi^3

We again use an explicit time stepping scheme for simplicity.

>>> D = 1.
>>> eq = (TransientTerm()
...       == DiffusionTerm(coeff=D)
...       + phase - phase**3)

>>> timeStepDuration = 0.1
>>> steps = 100
>>> if __name__ == '__main__':
...     viewer = Viewer(vars=(phase,), datamin=-1., datamax=1.)
...     for i in range(steps):
...         eq.solve(var=phase, dt=timeStepDuration)
...         viewer.plot()

"""
#!/usr/bin/env python
#!/usr/bin/env python

from fipy import CellVariable, Grid2D, Viewer, TransientTerm, DiffusionTerm

dx = dy = 0.5
nx = ny = 40
mesh = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)

concentration = CellVariable(name="concentration", mesh=mesh, value=0.)

concentration.constrain(1.0, where=mesh.facesBottom)
concentration.faceGrad[1].constrain(0.0, where=mesh.facesTop)
concentration.faceGrad[0].constrain(0.0, where=mesh.facesLeft)
concentration.faceGrad[0].constrain(0.0, where=mesh.facesRight)

D = 1.0
equation = TransientTerm() == DiffusionTerm(coeff=D)

timeStepDuration = 0.1
steps = 500

if __name__ == '__main__':
    viewer = Viewer(vars=(concentration,), datamin=0., datamax=1.)
    for i in range(steps):
        equation.solve(var=concentration, dt=timeStepDuration)
        viewer.plot()









