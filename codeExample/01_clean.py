import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

# Load the iris dataset
iris_bunch = load_iris()
# Handle the case where load_iris might return a tuple
if isinstance(iris_bunch, tuple):
    iris_data = iris_bunch[0]
else:
    iris_data = iris_bunch
    
df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)

# Add a duplicate row
df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

# Add an outlier row
df.loc[len(df)] = [1000, 1000, 1000, 1000]  # extreme values

# Detect duplicates
duplicates = df.duplicated()

# Detect outliers with IsolationForest
clf = IsolationForest(contamination=0.02, random_state=42)
outlier_preds = clf.fit_predict(df)
df['is_outlier'] = (outlier_preds == -1)

# Dimensionality reduction for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df.iloc[:, :4])
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

# Plot PCA result with outliers highlighted
plt.figure(figsize=(10, 6))
plt.scatter(df.loc[~df['is_outlier'], 'PCA1'], df.loc[~df['is_outlier'], 'PCA2'],
            c='blue', label='Normal', alpha=0.6)
plt.scatter(df.loc[df['is_outlier'], 'PCA1'], df.loc[df['is_outlier'], 'PCA2'],
            c='red', label='Outlier', marker='x', s=100)
plt.title('PCA of Iris Dataset with Outliers Highlighted')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend()
plt.grid(True)
plt.tight_layout()
curr_path= os.getcwd()


# Define the path to the 'data' directory relative to the current script
data_path = Path(curr_path).resolve() / 'data'

# Create the directory if it doesn't exist
data_path.mkdir(parents=True, exist_ok=True)
plot_name = "iris_outliers_pca.png"
plt_path = data_path / plot_name

plt.savefig(plt_path)
plt.close()

print(f"Plot saved to: {plt_path}")
