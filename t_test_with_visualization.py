import numpy as np ## for creating the samples
import scipy.stats as stats ## for the statitistical testing
import matplotlib.pyplot as plt ## for visualization
# Generate two groups of data
# np.random.seed(42)
n = 500 ## number of samples
group1 = np.random.normal(loc=10, scale=2, size=n)
group2 = np.random.normal(loc=9.9, scale=2, size=n)

# Plotting the distributions of the two groups
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.hist(group1, bins=15, alpha=0.5, label='Group 1')
plt.hist(group2, bins=15, alpha=0.5, label='Group 2')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Distributions of Two Groups')
plt.legend()

# Perform independent t-test
t_statistic, p_value = stats.ttest_ind(group1, group2)

# This is a test for the null hypothesis that 2 independent samples
# have identical average (expected) values. This test assumes that the
# populations have identical variances by default.


## Generate the t-distribution with appropriate degrees of freedom
df = len(group1) + len(group2) - 2
x = np.linspace(-5, 5, 1000)  # Range of values for the t-distribution
t_dist = stats.t.pdf(x, df)

## Plotting the t-distribution
plt.subplot(1, 2, 2)
plt.plot(x, t_dist, label='t-distribution')
plt.axvline(x=t_statistic, color='red', linestyle='--', label='t-statistic')

## Plotting the p-value as the area
p_value_area = stats.t.cdf(t_statistic, df)
plt.fill_between(x, 0, t_dist, where=np.abs(x) >= np.abs(t_statistic), color='red', alpha=0.3,
                 label='p-value')

## Plot annotations
plt.xlabel('t-value')
plt.ylabel('Probability Density')
plt.title('t-distribution with t-statistic and p-value')
plt.legend()

# Adjust the layout and display the plots
plt.tight_layout()

dpi = 500  # Adjust as needed
plt.savefig('output.png', dpi=dpi, bbox_inches='tight')

plt.show()
print("test statistic = ", t_statistic)
print("p-value = ", p_value)



if p_value < 0.05:
    print("Reject Null Hypothesis - Significant difference between two groups")
else:
    print("Accept Null Hypothesis - The difference is insignificant")

