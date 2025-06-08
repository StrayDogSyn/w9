import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression

# Set random seed for reproducibility
np.random.seed(42)

# Create 3 normally distributed variables
n = 500
x1 = np.random.normal(0, 1, n)
x2 = -x1 + np.random.normal(0, 0.5, n)  # correlated with x1
x3 = np.random.normal(0, 1, n)         # independent

# Create DataFrame
df_normal = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3})

# Compute covariance and correlation matrices
cov_matrix = df_normal.cov()
corr_matrix = df_normal.corr()

plt.figure(figsize=(14, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix (Normal)")
plt.tight_layout()
plt.show()

# Create more complicated non-linear dependence
z1 = np.random.uniform(-3, 3, n)
z2 = np.sin(z1) + 0.1 * np.random.normal(size=n)  # non-linear relationship


df_nonlin = pd.DataFrame({'z1': z1, 'z2': z2})

# Compute mutual information
mi = mutual_info_regression(df_nonlin[['z1']], df_nonlin['z2'])[0]

# Prepare figures
plt.figure(figsize=(14, 6))
sns.scatterplot(x=z1, y=z2)
plt.title(f"Nonlinear Relationship\nMutual Information: {mi:.2f}")
plt.xlabel("z1")
plt.ylabel("z2")
plt.tight_layout()
plt.show()
#######################
# Plot 2: Heatmap of covariance matrix
plt.subplot(1, 3, 2)
sns.heatmap(cov_matrix, annot=True, cmap="YlGnBu")
plt.title("Covariance Matrix (Normal)")

# Plot 3: Scatterplot of non-linear relation
plt.subplot(1, 3, 3)
sns.scatterplot(x=z1, y=z2)
plt.title(f"Nonlinear Relationship\nMutual Information: {mi:.2f}")
plt.xlabel("z1")
plt.ylabel("z2")

plt.tight_layout()
plt.show()
