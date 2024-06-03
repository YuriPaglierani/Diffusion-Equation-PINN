import os
import torch
import numpy as np
import fipy as fp
from torch.utils.data import TensorDataset
from multiprocessing import Pool
from typing import List, Tuple

# Define parameters
nx = ny = 50  # Number of grid points
dx = dy = 0.025  # Distance between grid points
D = 0.05  # Diffusivity
dt = 0.01  # Time step size
steps = 150

def simulate(temp: float) -> np.ndarray:
    """
    Simulate the diffusion process for a given boundary temperature.

    Args:
        temp (float): The boundary temperature.

    Returns:
        np.ndarray: Temperature distribution over time.
    """

    mesh = fp.Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)
    u = fp.CellVariable(name="T", mesh=mesh, hasOld=True)
    
    # Initial conditions
    u.setValue(0.0)
    # Set boundary conditions: fixed temperature on all boundaries
    u.constrain(temp, mesh.facesRight)
    u.constrain(temp, mesh.facesLeft)
    u.constrain(temp, mesh.facesTop)
    u.constrain(temp, mesh.facesBottom)

    # Define the equation
    eq = (fp.TransientTerm() == fp.DiffusionTerm(coeff=D))

    # Preallocate array to store the temperature distribution at each time step
    temperature_distribution = np.zeros((steps, nx * ny))

    # Solve over time and save each step
    for i in range(steps):
        u.updateOld() 
        eq.solve(var=u, dt=dt)
        temperature_distribution[i, :] = np.array(u.value)

    return temperature_distribution

# Function to handle multiprocessing
def run_simulation(temps: List[float]) -> Tuple[TensorDataset, TensorDataset, TensorDataset, float, torch.Tensor, torch.Tensor, TensorDataset]:
    """
    Run the simulation for multiple temperatures using multiprocessing.

    Args:
        temps (List[float]): List of temperatures to simulate.

    Returns:
        Tuple[TensorDataset, TensorDataset, TensorDataset, float, torch.Tensor, torch.Tensor, TensorDataset]: Datasets and statistics.
    """
    
    with Pool(processes=4) as pool:  # Adjust the number of processes based on your machine
        results = pool.map(simulate, temps)
    solutions = torch.zeros((len(results), steps, nx+2, ny+2))
    for i in range(len(results)):
        solutions[i, :, 1:-1, 1:-1] = torch.tensor(results[i]).reshape((steps, nx, ny))
    for i, temp in enumerate(temps):
        solutions[i, :, 0, :] = temp
        solutions[i, :, :, 0] = temp
        solutions[i, :, -1, :] = temp
        solutions[i, :, :, -1] = temp

    mesh = fp.Grid3D(dx=dx, dy=dy, dz=dt, nx=nx+2, ny=ny+2, nz=steps)
    x, y, t = mesh.cellCenters
    x, y, t = x.reshape(steps, nx+2, ny+2), y.reshape(steps, nx+2, ny+2), t.reshape(steps, nx+2, ny+2)
    data = np.zeros((steps, nx+2, ny+2, 3))
    data[:, :, :, 0] = x
    data[:, :, :, 1] = y
    data[:, :, :, 2] = t
    T = solutions
    dataset_start = np.concatenate([data, T[0].reshape(T[0].shape[0], T[0].shape[1], T[0].shape[2], 1)], axis=-1)
    dataset = dataset_start.reshape(-1, 4)

    total_samples = dataset.shape[0]
    train_size = int(0.8 * total_samples)
    validation_size = int(total_samples * 0.2 // 2)
    indices = np.random.permutation(total_samples)
    dataset_shuffled = dataset[indices]
    train_indices = indices[:train_size]
    validation_indices = indices[train_size:train_size+validation_size]
    test_indices = indices[train_size+validation_size:]

    train_data = dataset_shuffled[train_indices]
    validation_data = dataset_shuffled[validation_indices]
    test_data = dataset_shuffled[test_indices]

    mean_train = torch.tensor(train_data.mean(axis=0), dtype=torch.float32)
    std_train = torch.tensor(train_data.std(axis=0), dtype=torch.float32)

    train_data_tensor = torch.tensor(train_data, dtype=torch.float32)
    validation_data_tensor = torch.tensor(validation_data, dtype=torch.float32)
    test_data_tensor = torch.tensor(test_data, dtype=torch.float32)
    full_data_tensor = torch.tensor(dataset_start, dtype=torch.float32)

    # Create TensorDatasets
    train_dataset = TensorDataset(train_data_tensor)
    validation_dataset = TensorDataset(validation_data_tensor)
    test_dataset = TensorDataset(test_data_tensor)
    full_dataset = TensorDataset(full_data_tensor)

    return train_dataset, validation_dataset, test_dataset, D, mean_train, std_train, full_dataset

# Running the simulation
if __name__ == "__main__":
    if not os.path.exists('./data'):
        os.makedirs('./data')
    temperatures = np.array([100])  # Define temperature range
    train_dataset, validation_dataset, test_dataset, D, mean_train, std_train, full_dataset = run_simulation(temperatures)
    torch.save({'temperatures': temperatures,
                'train_set': train_dataset,
                'validation_set': validation_dataset,
                'test_set': test_dataset,
                'D': D,
                'mean_train': mean_train,
                'std_train': std_train}, './data/data.pth')
    torch.save({'full_dataset': full_dataset}, './data/full_dataset')
    