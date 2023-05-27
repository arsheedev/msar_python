import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

# Load the dataset from a CSV file
data = pd.read_csv('data.csv')
values = data['CAR'].values

# Fit the MSAR model
model = MarkovAutoregression(values, k_regimes=2, order=1)
result = model.fit()

# Predict the regime probabilities
regime_probs = result.filtered_marginal_probabilities[0]

# Plot the data and regime probabilities
plt.figure(figsize=(10, 6))
plt.plot(values, label='Data', color='blue')
plt.plot(regime_probs, label='Regime 0', color='red')
# plt.plot(1 - regime_probs, label='Regime 1', color='green')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Markov Switching Autoregressive Model')
plt.legend()
plt.show()
