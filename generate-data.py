import csv
import random

# Set random seed for reproducibility
random.seed(42)

# Define parameters
num_points = 100
transition_prob = 0.05  # Probability of switching states
mean1 = 5.0  # Mean of state 1
std1 = 1.0  # Standard deviation of state 1
mean2 = 10.0  # Mean of state 2
std2 = 2.0  # Standard deviation of state 2

# Generate the dataset
data = []
state = 1  # Initial state
for t in range(num_points):
    # Determine the current state
    if random.random() < transition_prob:
        state = 3 - state  # Switch between states 1 and 2

    # Generate value based on the current state
    if state == 1:
        value = random.gauss(mean1, std1)
    else:
        value = random.gauss(mean2, std2)

    # Add the data point to the dataset
    data.append((t, value))

# Save the dataset to a CSV file
with open('dataset.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['time', 'value'])
    writer.writerows(data)