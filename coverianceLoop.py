import numpy as np

# Sample Data (Columns are Features & Rows are Records)

data = np.array([
    [10, 15, 20, 25, 30],
    [12, 18, 21, 22, 28],
    [9, 14, 19, 26, 32],
    [11, 16, 22, 23, 27],
    [10, 17, 20, 24, 29]
])


# Finding Coverience Matrix Using Loop 
var_matrix = np.ones([5, 5])  # Initialize a 5x5 matrix
means = np.mean(data, axis=0) # Mean of Centralized Matrix (It can also be calculated using loop.)

for f1 in range(means.shape[0]):  # Feature 1
    for f2 in range(means.shape[0]):  # Feature 2
        s = 0  # Reset Sum
        for k in range(data.shape[0]):  # Loop Rows
            # Sum of (Product of Variances)
            s += ((data[k, f1] - mean[f1]) * (data[k, f2] - mean[f2]))
        # Dividing Sum(Variances) by (N-1)
        var_matrix[f1, f2] = s / (means.shape[0] - 1)

# Printing Output
print("Numpy Coverience Matrix:\n", var_matrix)

# Comparing Results With Built-in Function
print("Numpy Coverience Matrix:\n",np.cov(data, rowvar=False))
