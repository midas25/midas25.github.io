import numpy as np
import pandas as pd


def determine_true_false(value):
    if value <= 1.25:
        return False
    elif 1.25 < value <= 1.5:
        return np.random.rand() >= 0.95
    elif 1.5 < value <= 1.75:
        return np.random.rand() >= 0.85
    elif 1.75 < value <= 2.0:
        return np.random.rand() >= 0.8
    elif 2.0 < value <= 2.25:
        return np.random.rand() >= 0.15
    else:
        return True  # 100% True for values > 2.5


# Generate 10,000 random values between 1.0 and 5.0 rounded to 1 decimal place
np.random.seed()  # For reproducibility
random_values = np.round(np.random.uniform(1.0, 5.0, 100), 1)


# Apply the function to each value and store the results in a list as [value, result]
results = [[value, determine_true_false(value)] for value in random_values]
x_data = [results[i][0] for i in range(len(results))]
y_data = [results[i][1] for i in range(len(results))]
print(x_data)
print(y_data)

