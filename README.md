# Geometric Brownian Motion Simulator

This Python script simulates asset price paths using the Geometric Brownian Motion (GBM) model, a common stochastic process in financial modeling. It generates multiple price paths, visualizes them, and compares simulated results with theoretical expectations.

## Features

- Simulates GBM price paths with customizable parameters (starting price, volatility, drift, time, etc.).
- Plots individual price paths, mean, and 5th/95th percentiles.
- Visualizes the distribution of final prices against the theoretical lognormal distribution.
- Computes and compares theoretical vs. sample statistics (mean and variance).

## Requirements

- Python 3.x
- Libraries:
  - `numpy`
  - `matplotlib`
  - `scipy`

Install dependencies with:

```bash
pip install numpy matplotlib scipy