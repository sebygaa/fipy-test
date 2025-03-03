# %%
# Momemtum balance in advance

# %%
from fipy import (
    CellVariable, FaceVariable, Grid2D, DiffusionTerm, Viewer
)
from fipy.tools import numerix
from fipy.variables.faceGradVariable import _FaceGradVariable
from builtins import range

###############################################################################
# 1. Domain and discretization
###############################################################################
L = 100.0        # Domain extends from x=0..100, y=0..100
N = 100          # 100 cells in each direction
dL = L / N       # cell size = 1
viscosity = 1.0  # "nu" or dynamic viscosity for the diffusion term
U = 2.          # Inflow / outflow speed in the vertical direction
pressureRelaxation = 0.8
velocityRelaxation = 0.5

# Number of SIMPLE "sweeps" (iterations)
if __name__ == '__main__':
    sweeps = 300
else:
    sweeps = 5

mesh = Grid2D(nx=N, ny=N, dx=dL, dy=dL)

###############################################################################
# 2. Variables
###############################################################################
pressure = CellVariable(mesh=mesh, name='pressure', value=0.)
pressureCorrection = CellVariable(mesh=mesh, name='pCorrection', value=0.)
xVelocity = CellVariable(mesh=mesh, name='X velocity', value=0.)
yVelocity = CellVariable(mesh=mesh, name='Y velocity', value=0.)

# Face velocity for Rhie–Chow interpolation
velocity = FaceVariable(mesh=mesh, rank=1)

###############################################################################
# 3. Momentum (Stokes) Equations in SIMPLE form
###############################################################################
# Momentum eq for x-velocity:
#   nu * Laplacian(u) = dP/dx
#   => DiffusionTerm(coeff=viscosity) - pressure.grad.dot([1,0])
#
# Momentum eq for y-velocity:
#   nu * Laplacian(v) = dP/dy
#   => DiffusionTerm(coeff=viscosity) - pressure.grad.dot([0,1])
#
xVelocityEq = DiffusionTerm(coeff=viscosity) - pressure.grad.dot([1., 0.])
yVelocityEq = DiffusionTerm(coeff=viscosity) - pressure.grad.dot([0., 1.])

###############################################################################
# 4. Coefficients for Pressure Correction (SIMPLE)
###############################################################################
# ap holds diagonal of momentum matrix
ap = CellVariable(mesh=mesh, value=1.)

# "coeff" ~ 1/ap * faceArea * cellDistance => used in the pressure correction eq
coeff = (1. / ap.arithmeticFaceValue) * mesh._faceAreas * mesh._cellDistances
pressureCorrectionEq = DiffusionTerm(coeff=coeff) - velocity.divergence

###############################################################################
# 5. Boundary Conditions
###############################################################################
#
# Domain = 0 <= x <= 100, 0 <= y <= 100
# We'll define two "inlet" strips on the right side (x=100),
# each 5 units in height, at y in [90..95] and [70..75],
# with v=0.5, u=0 (flow into domain).
#
# Then two "outlet" strips on the left side (x=0),
# at y in [10..15] and [30..35], also with v=0.5, which is unusual physically,
# but meets the request for "two outlet flows at left-hand side, v=0.5".
#
# Everything else is no-slip (u=v=0).
###############################################################################

# Identify boundary faces
facesLeft   = mesh.facesLeft
facesRight  = mesh.facesRight
facesBottom = mesh.facesBottom
facesTop    = mesh.facesTop

Yface = mesh.faceCenters[1]

# Right boundary inlet segments:
rightInlet1 = facesRight & (Yface >= 90) & (Yface < 95)
rightInlet2 = facesRight & (Yface >= 70) & (Yface < 75)

# Left boundary outlet segments:
leftOutlet1 = facesLeft & (Yface >= 10) & (Yface < 15)
leftOutlet2 = facesLeft & (Yface >= 30) & (Yface < 35)

# 5a) X velocity constraints
# No slip by default
xVelocity.constrain(0., facesRight | facesLeft | facesBottom | facesTop)

# For the "inlets" and "outlets", the flow is purely vertical => xVelocity=0
# (already set by no slip or the above line—just reaffirming)
xVelocity.constrain(0., rightInlet1 | rightInlet2)
xVelocity.constrain(0., leftOutlet1 | leftOutlet2)

# 5b) Y velocity constraints
yVelocity.constrain(0., mesh.exteriorFaces)  # default no slip
# Right inlets => v=+0.5
yVelocity.constrain(U, rightInlet1)
yVelocity.constrain(U, rightInlet2)
# Left "outlets" => v=+0.5
yVelocity.constrain(U, leftOutlet1)
yVelocity.constrain(U, leftOutlet2)

# Optionally pin pressure somewhere to 0, e.g. bottom-left corner cell,
# but code doesn't do it by default.
# We'll just set zero correction at the bottom-left corner to avoid drifting:
Xface, Yface = mesh.faceCenters
bottomLeftCorner = facesLeft & facesBottom
pressureCorrection.constrain(0., bottomLeftCorner)

###############################################################################
# 6. SIMPLE Iteration
###############################################################################
if __name__ == '__main__':
    viewer = Viewer(vars=(pressure, xVelocity, yVelocity, velocity),
                    xmin=0., xmax=100., ymin=0., ymax=100.,
                    colorbar='vertical', scale=5)

for sweep in range(sweeps):
    # 6a) Momentum Solve for "starred" velocities
    xVelocityEq.cacheMatrix()
    xres = xVelocityEq.sweep(var=xVelocity, underRelaxation=velocityRelaxation)
    xmat = xVelocityEq.matrix

    yres = yVelocityEq.sweep(var=yVelocity, underRelaxation=velocityRelaxation)

    # 6b) Update ap from matrix diagonal
    ap[:] = -numerix.asarray(xmat.takeDiagonal())

    # 6c) Rhie–Chow face velocity correction
    presgrad = pressure.grad
    facepresgrad = _FaceGradVariable(pressure)
    volume = CellVariable(mesh=mesh, value=mesh.cellVolumes)
    contrvolume = volume.arithmeticFaceValue

    velocity[0] = (xVelocity.arithmeticFaceValue
                   + contrvolume / ap.arithmeticFaceValue
                     * (presgrad[0].arithmeticFaceValue - facepresgrad[0]))
    velocity[1] = (yVelocity.arithmeticFaceValue
                   + contrvolume / ap.arithmeticFaceValue
                     * (presgrad[1].arithmeticFaceValue - facepresgrad[1]))

    # Zero out face velocities on external boundaries except the inlets/outlets
    velocity[..., mesh.exteriorFaces.value] = 0.
    # Re-enforce the inlet/outlet velocity in the vertical direction at v=0.5
    velocity[1, rightInlet1.value] = U
    velocity[1, rightInlet2.value] = U
    velocity[1, leftOutlet1.value] = U
    velocity[1, leftOutlet2.value] = U

    # 6d) Pressure Correction
    pressureCorrectionEq.cacheRHSvector()
    pres = pressureCorrectionEq.sweep(var=pressureCorrection)
    rhs = pressureCorrectionEq.RHSvector

    # 6e) Update pressure, velocity with relaxation
    pressure.setValue(pressure + pressureRelaxation * pressureCorrection)
    xVelocity.setValue(
        xVelocity - (pressureCorrection.grad[0] / ap) * mesh.cellVolumes
    )
    yVelocity.setValue(
        yVelocity - (pressureCorrection.grad[1] / ap) * mesh.cellVolumes
    )

    if __name__ == '__main__':
        if sweep % 10 == 0:
            print('sweep:', sweep,
                  ', x residual:', xres,
                  ', y residual:', yres,
                  ', p residual:', pres,
                  ', continuity:', max(abs(rhs)))
viewer.plot()

# %%
# Delete the variables (for RAM)
# %%
del Xface, Yface,ap,bottomLeftCorner,coeff,contrvolume
del facepresgrad, facesBottom,facesLeft,facesRight, facesTop
del leftOutlet1, leftOutlet2, pressure, pressureCorrection
del pressureCorrectionEq, pressureRelaxation
del rightInlet1, rightInlet2, velocity

# %%
# Concentraiton Simulations

# %%

# YOU NEED the following variables: 
# xVelocity
# yVelocity
# uCell = xVelocity.value
# vCell = yVelocity.value

#!/usr/bin/env python

import fipy as fp
import numpy as np

###############################################################################
# 1. Domain: same 100x100 grid used for velocity
###############################################################################
nx, ny = 100, 100
Lx = Ly = 100.0
dx = dy = Lx / nx

mesh = fp.Grid2D(nx=nx, ny=ny, dx=dx, dy=dy)

###############################################################################
# 2. Import or define the final velocities from SIMPLE
###############################################################################
# Here we assume you have arrays "uData" and "vData" that correspond
# to the converged xVelocity, yVelocity (cell-centered) from the momentum step.
# For illustration, we'll just show placeholders:
# (In a real code, you'd either pass them directly, or load from a file.)

#uData = np.zeros(nx*ny)  # Placeholders: In reality, fill with your final xVelocity
#vData = np.zeros(nx*ny)  # Placeholders: final yVelocity
uData = xVelocity.value.reshape(100,100)
vData = yVelocity.value.reshape(100,100)

# e.g. uData[i] = xVelocity.value[i] from your momentum code.

# Build FiPy CellVariables from these arrays
uCell = fp.CellVariable(mesh=mesh, value=uData, name="u-vel")
vCell = fp.CellVariable(mesh=mesh, value=vData, name="v-vel")

###############################################################################
# 3. Define the species concentration
###############################################################################
concentration = fp.CellVariable(mesh=mesh, name="solventConcentration", value=0.0)

# You might do a "solvent mass fraction" or something similar. We'll keep it simple.

###############################################################################
# 4. PDE Coefficients
###############################################################################
D = 1.0e-3  # diffusion coefficient (example)

###############################################################################
# 5. Boundary Conditions
# 
# We'll approximate:
#   - Right top inlets => c=0 (pure hot air, zero solvent).
#   - Bottom => c=1.0, representing a saturated film
#   - Left outlets => zero gradient (outflow).
#   - Everything else => no flux.
###############################################################################

# Identify boundary faces
facesLeft   = mesh.facesLeft
facesRight  = mesh.facesRight
facesBottom = mesh.facesBottom
facesTop    = mesh.facesTop

Yfaces = mesh.faceCenters[1]

# Right-top inlets (two segments):
# e.g. [70..75], [90..95] from your velocity scenario
inlet1 = facesRight & (Yfaces>=70) & (Yfaces<75)
inlet2 = facesRight & (Yfaces>=90) & (Yfaces<95)

# We set c=0 on these inlets
concentration.constrain(0.0, where=inlet1)
concentration.constrain(0.0, where=inlet2)

# Bottom => c=1, to represent film source
concentration.constrain(1.0, where=facesBottom)

# Left outlets => zero gradient => typical outflow
# In FiPy, that means do NOT set a Dirichlet constraint. We'll do:
concentration.faceGrad[0].constrain(0.0, where=facesLeft)

# Top => no flux => zero gradient in y
concentration.faceGrad[1].constrain(0.0, where=facesTop)

# The remaining part of the right boundary not in [70..75] or [90..95]
# you might also want no flux or zero gradient, e.g.:
noFluxRight = facesRight & ~inlet1 & ~inlet2
concentration.faceGrad[0].constrain(0.0, where=noFluxRight)

###############################################################################
# 6. Define the Convection–Diffusion Equation
###############################################################################
#   d(c)/dt + div(u*c) = div(D grad(c))
# We'll define the velocity as a tuple of cell data for ConvectionTerm.
# FiPy expects something like ConvectionTerm(coeff=(uCell, vCell)).

eq = (
    fp.TransientTerm(var=concentration)
    + fp.ConvectionTerm(coeff=(uCell, vCell), var=concentration)
    == fp.DiffusionTerm(coeff=D, var=concentration)
)

###############################################################################
# 7. Solve Transiently
###############################################################################
timeStep = 1.0
nSteps   = 200

if __name__ == "__main__":
    viewer = fp.Viewer(vars=(concentration,), datamin=0., datamax=1.0)
    for step in range(nSteps):
        eq.solve(dt=timeStep)

    viewer.plot()
    print("Time step:", step)

# %%
