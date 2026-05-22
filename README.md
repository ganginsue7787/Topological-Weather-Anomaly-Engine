# Topological Weather Anomaly Engine

Persistent topology-based atmospheric anomaly detection using ERA5 vorticity fields.

## Overview

This project explores topology-aware anomaly detection on atmospheric vortex structures
using ERA5 reanalysis data and persistent topology-inspired geometric metrics.

The framework computes:

- Vorticity fields from ERA5 u/v wind vectors
- Anisotropic geometric distances
- MST-based topological persistence indicators
- Topological Anomaly Score (TAS)

## Core Equations

### Vorticity

ω = ∂x v - ∂y u

### Anisotropic Metric

d = sqrt(d_space^2 + α^2(ω_i - ω_j)^2)

## Features

- ERA5 NetCDF support
- Real vorticity field computation
- Topological anomaly scoring
- Typhoon structure analysis
- Visualization pipeline

## Suggested Dataset

ERA5 hourly data on single levels:
https://cds.climate.copernicus.eu/

Recommended variables:
- 10m_u_component_of_wind
- 10m_v_component_of_wind
- mean_sea_level_pressure

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python src/tas_engine.py
```

## Repository Structure

```text
topological-weather-anomaly-engine/
├── data/
├── docs/
├── notebooks/
├── outputs/
├── src/
├── README.md
├── requirements.txt
└── LICENSE
```

## License

MIT License