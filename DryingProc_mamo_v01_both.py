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
#!/usr/bin/env python

import fipy as fp
import numpy as np
from fipy.tools import numerix
from fipy.variables.faceGradVariable import _FaceGradVariable

###############################################################################
# Mesh: 100×100
###############################################################################
nx = ny = 100
dx = dy = 1.0
mesh = fp.Grid2D(nx=nx, ny=ny, dx=dx, dy=dy)

###############################################################################
# Momentum variables (SIMPLE)
###############################################################################
pressure   = fp.CellVariable(mesh=mesh, name="pressure", value=0.0)
pCorr      = fp.CellVariable(mesh=mesh, name="pCorr",    value=0.0)
xVelocity  = fp.CellVariable(mesh=mesh, name="xVelocity", value=0.0)
yVelocity  = fp.CellVariable(mesh=mesh, name="yVelocity", value=0.0)

# FaceVariable used only internally for Rhie–Chow
faceVelocity = fp.FaceVariable(mesh=mesh, rank=1)

# Momentum eq (Stokes)
viscosity = 1.0
xMomEq = fp.DiffusionTerm(coeff=viscosity) - pressure.grad.dot([1.,0.])
yMomEq = fp.DiffusionTerm(coeff=viscosity) - pressure.grad.dot([0.,1.])

ap = fp.CellVariable(mesh=mesh, value=1.0)
coeff = (1. / ap.arithmeticFaceValue) * mesh._faceAreas * mesh._cellDistances
pCorrEq = fp.DiffusionTerm(coeff=coeff) - faceVelocity.divergence

pressureRelaxation = 0.8
velocityRelaxation = 0.5

###############################################################################
# Momentum boundary conditions example
# Let's pick some generic BC: no-slip all around except a "top-lid" v=0, u=1.0 
###############################################################################
facesLeft   = mesh.facesLeft
facesRight  = mesh.facesRight
facesBottom = mesh.facesBottom
facesTop    = mesh.facesTop

# Default: no-slip
xVelocity.constrain(0.0, mesh.exteriorFaces)
yVelocity.constrain(0.0, mesh.exteriorFaces)

# For demonstration: top boundary is a "moving lid" in x
xVelocity.constrain(1.0, facesTop)
yVelocity.constrain(0.0, facesTop)

###############################################################################
# SIMPLE iteration
###############################################################################
def solveMomentum(nSweeps=50):
    for sweep in range(nSweeps):
        # (a) Solve x-momentum
        xMomEq.cacheMatrix()
        xRes = xMomEq.sweep(var=xVelocity, underRelaxation=velocityRelaxation)
        xMat = xMomEq.matrix

        # (b) Solve y-momentum
        yRes = yMomEq.sweep(var=yVelocity, underRelaxation=velocityRelaxation)

        # (c) Update ap from x-momentum diagonal
        ap[:] = -numerix.asarray(xMat.takeDiagonal())

        # (d) Rhie–Chow face velocity
        presGrad = pressure.grad
        facePresGrad = _FaceGradVariable(pressure)

        cellVol       = fp.CellVariable(mesh=mesh, value=mesh.cellVolumes)
        contrVol      = cellVol.arithmeticFaceValue

        faceVelocity[0] = (
            xVelocity.arithmeticFaceValue 
            + contrVol / ap.arithmeticFaceValue 
              * (presGrad[0].arithmeticFaceValue - facePresGrad[0])
        )
        faceVelocity[1] = (
            yVelocity.arithmeticFaceValue
            + contrVol / ap.arithmeticFaceValue
              * (presGrad[1].arithmeticFaceValue - facePresGrad[1])
        )

        # Zero out external faces
        faceVelocity[..., mesh.exteriorFaces.value] = 0.

        # (e) Pressure correction
        pCorrEq.cacheRHSvector()
        pRes = pCorrEq.sweep(var=pCorr)
        rhs  = pCorrEq.RHSvector

        # (f) Update p, velocity
        pressure.setValue(pressure + pressureRelaxation * pCorr)
        xVelocity.setValue(
            xVelocity - (pCorr.grad[0]/ap)* mesh.cellVolumes
        )
        yVelocity.setValue(
            yVelocity - (pCorr.grad[1]/ap)* mesh.cellVolumes
        )

        if sweep % 10 == 0:
            print(f"SIMPLE sweep={sweep}, xRes={xRes}, yRes={yRes}, "
                  f"pRes={pRes}, continuity={max(abs(rhs))}")

###############################################################################
# Mass transport variables (Convection–Diffusion)
###############################################################################
species = fp.CellVariable(mesh=mesh, name="species", value=0.0)
D = 1e-3

# Example BC: bottom = species=1, top=0
species.constrain(1.0, facesBottom)
species.constrain(0.0, facesTop)

# left & right => zero gradient
species.faceGrad[0].constrain(0.0, facesLeft)
species.faceGrad[0].constrain(0.0, facesRight)

massEq = (
    fp.TransientTerm(var=species)
    + fp.ConvectionTerm(coeff=(xVelocity, yVelocity), var=species)
    == fp.DiffusionTerm(coeff=D, var=species)
)

###############################################################################
# Main
###############################################################################
if __name__ == "__main__":

    print("Solving momentum...")
    solveMomentum(nSweeps=50)  # adjust as needed

    print("Solving mass transport...")
    dt = 1.0
    steps = 50

    viewer = fp.Viewer(vars=(species,), datamin=0., datamax=1.)
    for step in range(steps):
        massEq.solve(dt=dt)
        if step % 10 == 0:
            print(f"Mass step={step}")
            viewer.plot()



# %%
