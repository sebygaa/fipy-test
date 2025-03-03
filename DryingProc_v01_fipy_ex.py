# %%
#!/usr/bin/env python

from fipy import CellVariable, Grid2D, Viewer
from fipy.terms import TransientTerm, DiffusionTerm, ConvectionTerm

# --- 1. Grid Setup (20 x 20 mesh) ---
nx = ny = 100
dx = dy = 0.2
mesh = Grid2D(nx=nx, ny=ny, dx=dx, dy=dy)

# --- 2. Define the Concentration Variable ---
c = CellVariable(name="concentration", mesh=mesh, value=0.0)

# --- 3. Physical Parameters ---
D = 1.0                       # Diffusion coefficient
u = (-0.1, 0.0)               # Velocity vector (flow from right to left)

# --- 4. Boundary Conditions ---
# Bottom boundary => Dirichlet c = 1
c.constrain(1.0, where=mesh.facesBottom)

# Top boundary => no flux in y => faceGrad[1] = 0
c.faceGrad[1].constrain(0.0, where=mesh.facesTop)

# Left boundary => no diffusion flux in x => faceGrad[0] = 0
c.faceGrad[0].constrain(0.0, where=mesh.facesLeft)

# Right boundary => no diffusion flux in x => faceGrad[0] = 0
c.faceGrad[0].constrain(0.0, where=mesh.facesRight)

# --- 5. PDE: Transient Convection–Diffusion ---
#     d(c)/dt + (u·grad(c)) = D * Laplacian(c)
equation = (
    TransientTerm(var=c)
    + ConvectionTerm(coeff=u, var=c)
    == DiffusionTerm(coeff=D, var=c)
)

# --- 6. Time-stepping ---
dt = 0.1    # time step
steps = 100

if __name__ == "__main__":
    viewer = Viewer(vars=(c,), datamin=0., datamax=1.)

    for step in range(steps):
        equation.solve(var=c, dt=dt)
        viewer.plot()

# %%
