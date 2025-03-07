import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("ai4i2020.csv")

# Set Seaborn style
sns.set(style="whitegrid")

# Create scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df["Rotational speed [rpm]"], y=df["Tool wear [min]"], color="blue", alpha=0.6)

# Labels and Title
plt.xlabel("Rotational Speed (rpm)")
plt.ylabel("Tool Wear (min)")
plt.title("⚙️ Relationship Between Rotational Speed and Tool Wear")

# Show plot
plt.show()
