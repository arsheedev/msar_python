import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
data = pd.read_csv('data.csv')

# Define the MSAR model
model = sm.tsa.MarkovAutoregression(data['ROA'], k_regimes=2, order=1)

# Estimate the model parameters
model_result = model.fit()

# Predict model that have been estimated
predictions = model_result.predict()

# Analyze the model results
print(predictions)
print(model_result.summary())
print(model_result.filtered_marginal_probabilities[0])

# Show Chart
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(data['ROA'], label='Tahun-Bulan')
ax.plot(model_result.filtered_marginal_probabilities[0], label='Regime 1')
ax.plot(predictions, label='Predictions')
ax.legend()
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Markov Switching Autoregressive (MSAR) Model')
plt.show()