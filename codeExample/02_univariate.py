import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm, skewnorm, t

np.random.seed(42)

# Generate samples  
normal = np.array(norm.rvs(size=1000))
skewed = np.array(skewnorm.rvs(a=10, size=1000))     # a > 0 → right skew
heavy = np.array(t.rvs(df=2, size=1000))             # df small → heavy tails

# Plot
plt.figure(figsize=(12, 4))

for i, (dataset, label) in enumerate([(normal, "Normal"), 
                                   (skewed, "Skewed"), 
                                   (heavy, "Heavy-Tailed")]):
    plt.subplot(1, 3, i+1)
    sns.histplot(dataset, kde=True, stat='density', bins=30)
    plt.title(label)
    plt.xlabel("Value")
    plt.ylabel("Density")

plt.tight_layout()
plt.show()

## Box Plot Normal vs Skewed
plt.figure(figsize=(12, 4))

for i, (dataset, label) in enumerate([(normal, "Normal"), 
                                   (skewed, "Skewed")]):
    plt.subplot(1, 2, i+1)
    sns.boxplot(y=dataset)
    plt.title(label)
    plt.xlabel(label)
    plt.ylabel("quantile")
plt.tight_layout()
plt.show()
    
## QQ Plot Normal vs Heavy

from scipy import stats

#import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
for i, (dataset, label) in enumerate([(normal, "Normal"), 
                                   (heavy, "Heavy")]):
    plt.subplot(1, 2, i+1)
    stats.probplot(dataset, dist="norm", plot=plt)
    plt.title("Q-Q Plot-" + label)
plt.tight_layout()
plt.show()



#test_data = normal
test_data = skewed

show_normal = False 
if show_normal:
    test_data = normal
else:
    test_data = skewed


from scipy.stats import normaltest
stat, p = normaltest(test_data)
print(f"K² Statistic: {stat:.4f}, p-value: {p:.4f}")

show_normal = False 
if show_normal:
    test_data = normal
else:
    test_data = heavy

from scipy.stats import jarque_bera
stat, p = jarque_bera(test_data)
print(f"JB Statistic: {stat:.4f}, p-value: {p:.4f}")



