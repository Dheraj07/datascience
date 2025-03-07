import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("ai4i2020.csv")

# Set Seaborn stylef
sns.set(style="whitegrid")

# Create a figure with multiple histograms
fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 3 columns

# 1️⃣ **Histogram for Tool Wear**
sns.histplot(df["Tool wear [min]"], bins=30, kde=True, color="blue", ax=axes[0])
axes[0].set_xlabel("Tool Wear [min]")
axes[0].set_ylabel("Frequency")
axes[0].set_title("📊 Distribution of Tool Wear")

# 2️⃣ **Histogram for Process Temperature**
sns.histplot(df["Process temperature [K]"], bins=30, kde=True, color="red", ax=axes[1])
axes[1].set_xlabel("Process Temperature [K]")
axes[1].set_ylabel("Frequency")
axes[1].set_title("📊 Distribution of Process Temperature")

# 3️⃣ **Histogram for Rotational Speed**
sns.histplot(df["Rotational speed [rpm]"], bins=30, kde=True, color="green", ax=axes[2])
axes[2].set_xlabel("Rotational Speed [rpm]")
axes[2].set_ylabel("Frequency")
axes[2].set_title("📊 Distribution of Rotational Speed")

# Adjust layout and show plots
plt.tight_layout()
plt.show()
