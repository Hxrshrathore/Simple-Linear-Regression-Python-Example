import numpy as np
import pandas as pd

# Create sample data
X = np.linspace(0, 10, 100)  # Input data
y = 2 * X + 1 + 69 + np.random.normal(0, 1, 100)  # Output data with some noise

# Create a DataFrame
data = pd.DataFrame({'X': X, 'y': y})

# Save the dataset as a CSV file
data.to_csv('data/dataset.csv', index=False)
