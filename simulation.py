import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

# Parameters
s_0 = 131.00        # Starting Price
sigma = 0.6         # Volatility (Annualized)
mu = 0.25           # Drift rate
years = 5           # Total time in years
steps_per_year = 252  # Number of steps per year (trading days in a year)
paths = 500         # Number of simulations
seed = 42           # Random seed for reproducibility

# Derived parameters
time = int(years * steps_per_year)  # Total number of steps
delta = 1.0 / steps_per_year        # Time step size
time_array = np.arange(0, time + 1) * delta  # Time array in years

def wiener_process(delta, sigma, time, paths, seed=None):
    """Generate increments of the Wiener process."""
    rng = np.random.default_rng(seed)
    return sigma * rng.normal(loc=0, scale=np.sqrt(delta), size=(time, paths))

def gbm_returns(delta, sigma, time, mu, paths, seed=None):
    """Generate GBM returns."""
    process = wiener_process(delta, sigma, time, paths, seed=seed)
    return np.exp(process + (mu - sigma**2 / 2) * delta)

def gbm_levels(s_0, delta, sigma, time, mu, paths, seed=None):
    """Generate GBM price levels."""
    returns = gbm_returns(delta, sigma, time, mu, paths, seed=seed)
    stacked = np.vstack([np.ones(paths), returns])
    return s_0 * stacked.cumprod(axis=0)

# Generate price paths
price_paths = gbm_levels(s_0, delta, sigma, time, mu, paths, seed=seed)

# Plot simulated GBM paths
plt.figure(figsize=(10, 6))
plt.plot(time_array, price_paths, linewidth=0.4, alpha=0.7)
plt.title("Simulated GBM Paths")
plt.xlabel("Time (years)")
plt.ylabel("Price")
plt.show()

# Compute and plot mean and percentiles
mean_prices = np.mean(price_paths, axis=1)
percentile_5 = np.percentile(price_paths, 5, axis=1)
percentile_95 = np.percentile(price_paths, 95, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(time_array, mean_prices, 'k-', label='Mean')
plt.plot(time_array, percentile_5, 'b--', label='5th Percentile')
plt.plot(time_array, percentile_95, 'r--', label='95th Percentile')
plt.legend()
plt.title("Price Paths Statistics")
plt.xlabel("Time (years)")
plt.ylabel("Price")
plt.show()

# Final prices analysis
final_prices = price_paths[-1, :]

# Theoretical distribution
t = years
s = sigma * np.sqrt(t)
scale = s_0 * np.exp((mu - sigma**2 / 2) * t)
x = np.linspace(np.min(final_prices), np.max(final_prices), 100)
pdf = lognorm.pdf(x, s, scale=scale)

plt.figure(figsize=(10, 6))
plt.hist(final_prices, bins=30, density=True, alpha=0.5, label='Simulated')
plt.plot(x, pdf, 'r-', label='Theoretical')
plt.legend()
plt.title("Distribution of Final Prices")
plt.xlabel("Price")
plt.ylabel("Density")
plt.show()

# Print statistics
theoretical_mean = s_0 * np.exp(mu * t)
theoretical_var = (np.exp(sigma**2 * t) - 1) * s_0**2 * np.exp(2 * mu * t)
sample_mean = np.mean(final_prices)
sample_var = np.var(final_prices)
print(f"Theoretical Mean: {theoretical_mean:.2f}, Sample Mean: {sample_mean:.2f}")
print(f"Theoretical Variance: {theoretical_var:.2f}, Sample Variance: {sample_var:.2f}")