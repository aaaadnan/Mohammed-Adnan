# Mohammed-Adnan
data analytics project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Age groups and their frequencies
age_groups = [35, 45, 55, 65, 70, 75]
age_distribution = [0.20, 0.25, 0.20, 0.15, 0.10, 0.10]  # Uneven distribution for realism

# Generate age data
ages = np.random.choice(age_groups, size=60000, p=age_distribution)

# Define weighted purchase probabilities by age
weighted_probs = {35: 0.7, 45: 0.6, 55: 0.4, 65: 0.25, 70: 0.15, 75: 0.1}

# Assign purchases using weighted probabilities
purchases_weighted = [np.random.rand() < weighted_probs[age] for age in ages]

# Create DataFrame for weighted model
df_weighted = pd.DataFrame({'Age': ages, 'Purchased': purchases_weighted})

# Unweighted model (equal probability for all age groups)
uniform_prob = 0.4
purchases_unweighted = [np.random.rand() < uniform_prob for _ in ages]

df_unweighted = pd.DataFrame({'Age': ages, 'Purchased': purchases_unweighted})

# Conditional Probability Calculation
def conditional_prob(df, age_value):
    subset = df[df['Age'] == age_value]
    if len(subset) == 0:
        return 0
    return subset['Purchased'].mean()

# Calculate conditional probabilities for age 45
cp_weighted = conditional_prob(df_weighted, 45)
cp_unweighted = conditional_prob(df_unweighted, 45)

print(f"Weighted P(Purchase | Age=45): {cp_weighted:.4f}")
print(f"Unweighted P(Purchase | Age=45): {cp_unweighted:.4f}")

# Visualization
agewise_cp_weighted = df_weighted.groupby('Age')['Purchased'].mean()
agewise_cp_unweighted = df_unweighted.groupby('Age')['Purchased'].mean()

plt.figure(figsize=(10, 6))
plt.plot(agewise_cp_weighted.index, agewise_cp_weighted.values, marker='o', label='Weighted Model', color='blue')
plt.plot(agewise_cp_unweighted.index, agewise_cp_unweighted.values, marker='s', label='Unweighted Model', color='green')
plt.title('Conditional Probability of Purchase by Age Group')
plt.xlabel('Age')
plt.ylabel('P(Purchase | Age)')
plt.legend()
plt.grid(True)
plt.show()
