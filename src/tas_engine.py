import numpy as np
import xarray as xr

def compute_vorticity(u, v, dx, dy):
    dvdx = np.gradient(v, axis=-1) / dx
    dudy = np.gradient(u, axis=-2) / dy
    return dvdx - dudy

def tas_score(values):
    mu = np.mean(values)
    sigma = np.std(values)
    return (values - mu) / sigma

if __name__ == "__main__":
    print("Topological Weather Anomaly Engine initialized.")