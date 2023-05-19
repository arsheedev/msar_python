import pandas as pd
import statsmodels.api as sm

# Load the CSV file into a DataFrame
df = pd.read_csv('dataset.csv')

# Define the MSAR model
model = sm.tsa.MarkovAutoregression(df['value'], k_regimes=2, order=1)

# Estimate the model parameters
model_result = model.fit()

# Predict model that have been estimated
predictions = model_result.predict()

# Analyze the model results
print(predictions)
print(model_result.summary())