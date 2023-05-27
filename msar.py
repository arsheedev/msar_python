import pandas as pd
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt


# Load data from CSV file
data = pd.read_csv('data.csv')
observations = data['Short_Term_Mistmach'].dropna().values

# Define the MSAR model
with pm.Model() as model:
    # Priors for model parameters
    sigma1 = pm.Exponential('sigma1', 1.0)
    sigma2 = pm.Exponential('sigma2', 1.0)
    p = pm.Dirichlet('p', a=np.array([1, 1]))
    state = pm.Categorical('state', p=p, shape=len(observations))
    
    # Define autoregressive component
    coeff1 = pm.Normal('coeff1', mu=0, sd=10)
    coeff2 = pm.Normal('coeff2', mu=0, sd=10)
    rho1 = pm.Normal('rho1', mu=0, sd=1)
    rho2 = pm.Normal('rho2', mu=0, sd=1)
    ar1 = pm.AR('ar1', rho=rho1, shape=len(observations))
    ar2 = pm.AR('ar2', rho=rho2, shape=len(observations))
    mu = coeff1 * ar1 + coeff2 * ar2
    
    # Model likelihood
    obs_sigma = pm.math.switch(state, sigma1, sigma2)
    obs = pm.Normal('obs', mu=mu, sd=obs_sigma, observed=observations)
    
    # Inference
    trace = pm.sample(2000, tune=1000, chains=2)
    pm.traceplot(trace)

# Generate predictions
state_pred = trace['state'].mean(axis=0)
ar1_pred = trace['ar1'].mean(axis=0)
ar2_pred = trace['ar2'].mean(axis=0)
coeff1_pred = trace['coeff1'].mean()
coeff2_pred = trace['coeff2'].mean()
predictions = coeff1_pred * ar1_pred + coeff2_pred * ar2_pred

# Plot the data and predictions
plt.figure(figsize=(12, 6))
plt.plot(range(len(observations)), observations, label='Data')
plt.plot(range(len(predictions)), predictions, label='Predictions', color='red')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
