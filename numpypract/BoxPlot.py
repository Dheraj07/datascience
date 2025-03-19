import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Load dataset (Update file path if needed)
df = pd.read_csv("ai4i2020.csv")

# Selecting a categorical variable ('Type') and a numerical variable ('Air temperature [K]')
grouped_data = [df[df["Type"] == t]["Air temperature [K]"] for t in df["Type"].unique()]

# Normality Check (Shapiro-Wilk Test)
normality_results = [stats.shapiro(group.dropna())[1] for group in grouped_data]
if all(p > 0.05 for p in normality_results):
    print("Normality assumption holds (p > 0.05).")
else:
    print("Data is not normally distributed.")

# Homogeneity of Variance Check (Levene’s Test)
levene_stat, levene_p = stats.levene(*grouped_data)
if levene_p > 0.05:
    print("Equal variance assumption holds (Levene’s p > 0.05).")
else:
    print("Variances are not equal (Consider Welch’s ANOVA).")

# One-Way ANOVA
f_stat, p_value = stats.f_oneway(*grouped_data)
print(f"ANOVA F-statistic: {f_stat:.4f}, p-value: {p_value:.4f}")

# If ANOVA is significant, perform Tukey's HSD test
if p_value < 0.05:
    tukey = pairwise_tukeyhsd(df["Air temperature [K]"], df["Type"])
    print("\nPost-Hoc Analysis (Tukey HSD):")
    print(tukey)
else:
    print("No significant difference found between groups (p > 0.05).")

# Visualization - Box Plot
sns.set(style="whitegrid")  # Set style for better visualization

plt.figure(figsize=(8, 5))
sns.boxplot(x="Type", y="Air temperature [K]", data=df, palette="Set2")

# Add title and labels
plt.title("Comparison of Air Temperature Across Machine Types", fontsize=14)
plt.xlabel("Machine Type", fontsize=12)
plt.ylabel("Air Temperature (K)", fontsize=12)

# Show plot
plt.show()
