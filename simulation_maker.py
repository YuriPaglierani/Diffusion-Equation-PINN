import fipy as fp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Tuple, Callable

def setup_simulation(nx: int, 
                     ny: int, 
                     dx: float, 
                     dy: float, 
                     D: float) -> Tuple:
    """
    Set up the FiPy simulation with initial and boundary conditions.
    
    Args:
        nx (int): Number of grid points in the x direction.
        ny (int): Number of grid points in the y direction.
        dx (float): Grid spacing in the x direction.
        dy (float): Grid spacing in the y direction.
        D (float): Diffusion coefficient.

    Returns:
        Tuple: The mesh, temperature variable, and diffusion equation.
    """

    # Define the domain size
    Lx, Ly = nx * dx, ny * dy

    # Create mesh and variables
    mesh = fp.Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)
    u = fp.CellVariable(name="temperature", mesh=mesh, hasOld=True)

    # Initial conditions and boundary conditions
    u.setValue(0)
    u.constrain(100.0, mesh.facesTop)
    u.constrain(100.0, mesh.facesBottom)
    u.constrain(100.0, mesh.facesLeft)
    u.constrain(100.0, mesh.facesRight)

    # Define the diffusion equation
    eq = (fp.TransientTerm() == fp.DiffusionTerm(coeff=D))

    return mesh, u, eq

def run_simulation(u: fp.CellVariable, 
                   eq, 
                   frames: int, 
                   dt: float) -> Callable[[int], Tuple[plt.contour.QuadContourSet]]:
    """
    Run the simulation and update the plot for each frame.
    
    Args:
        u (fp.CellVariable): The variable representing temperature.
        eq: The diffusion equation to solve.
        frames (int): Number of frames in the animation.
        dt (float): Time step size.
        nx (int): Number of grid points in the x direction.
        ny (int): Number of grid points in the y direction.
        x (np.ndarray): x-coordinates of the mesh.
        y (np.ndarray): y-coordinates of the mesh.
        levels (np.ndarray): Contour levels for the plot.
        ax (plt.Axes): The axes object for the plot.

    Returns:
        Callable[[int], Tuple[plt.contour.QuadContourSet]]: The update function for the animation.
    """

    def update(frame: int) -> Tuple[plt.contour.QuadContourSet]:
        u.updateOld()
        eq.solve(var=u, dt=dt)
        u_reshaped = u.value.reshape((ny, nx))
        ax.clear()
        contour = ax.contourf(x, y, u_reshaped, levels=levels, cmap=plt.cm.jet, vmin=0, vmax=100)
        ax.set_title("Temperature Distribution")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        return contour,

    return update

if __name__ == "__main__":
    # Simulation parameters
    nx, ny = 50, 50
    dx, dy = 0.025, 0.025
    dt = 0.01
    D = 0.05
    frames = 150

    # Setup the simulation
    mesh, u, eq = setup_simulation(nx, ny, dx, dy, D)

    # Prepare the plot
    fig, ax = plt.subplots()
    levels = np.linspace(0, 100, num=21)
    x, y = mesh.cellCenters
    x, y = x.reshape((ny, nx)), y.reshape((ny, nx))

    # Initial plot
    u_reshaped = u.value.reshape((ny, nx))
    contour = ax.contourf(x, y, u_reshaped, levels=levels, cmap=plt.cm.jet, vmin=0, vmax=100)
    cbar = fig.colorbar(contour, ax=ax)
    ax.set_title("Temperature Distribution")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Create and run the animation
    update_func = run_simulation(u, eq, frames, dt)
    ani = FuncAnimation(fig, update_func, frames=frames, blit=False, repeat=False)
    ani.save('/simulations/benchmark.gif', writer='pillow', fps=10)

    # Show plot in window (remove this line if running on a server without a display)
    plt.show()