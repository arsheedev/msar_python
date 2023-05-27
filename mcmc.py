import numpy as np
import pymc3 as pm
import pandas as pd
import matplotlib.pyplot as plt


# Load data from CSV file
data = pd.read_csv('data.csv')

# Extract the relevant column from the DataFrame
observations = data['Short_Term_Mistmach'].dropna().values

# Model specification
with pm.Model() as model:
    # Priors
    p = pm.Beta("p", alpha=1, beta=1, shape=2)  # Transition probabilities
    mu = pm.Normal("mu", mu=0, sigma=10, shape=2)  # Regime means
    sigma = pm.HalfNormal("sigma", sigma=10, shape=2)  # Regime standard deviations

    # Hidden states
    states = pm.Categorical("states", p=p, shape=len(observations))

    # Observations
    obs = pm.Normal("obs", mu=mu[states], sigma=sigma[states], observed=observations)

    # Inference
    trace = pm.sample(2000, tune=1000, chains=2)

# Generate predictions
n_predictions = 100  # Number of predictions to generate

# Sample from the posterior distribution of hidden states
posterior_states = pm.sample_posterior_predictive(trace, model=model, samples=n_predictions)

# Get the mean predicted observation for each sample
predicted_observations = posterior_states["obs"].mean(axis=0)

# Plot original observations and predicted values
plt.figure()
plt.plot(np.arange(len(observations)), observations, label="Data Asli")
plt.plot(predicted_observations, label="Prediksi")
plt.title('MSAR Grafik Short_Term_Mistmach')
plt.legend()
plt.show()