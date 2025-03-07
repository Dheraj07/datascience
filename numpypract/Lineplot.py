import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("ai4i2020.csv")

# Create a time index for trend visualization
df["time_index"] = range(len(df))

# Set Seaborn style
sns.set(style="whitegrid")

# Create the figure
plt.figure(figsize=(12, 6))

# 1️⃣ **Plot Sensor Trends Over Time**
sns.lineplot(x=df["time_index"], y=df["Tool wear [min]"], label="Tool Wear", color="blue")
sns.lineplot(x=df["time_index"], y=df["Process temperature [K]"], label="Process Temperature", color="red")
sns.lineplot(x=df["time_index"], y=df["Rotational speed [rpm]"], label="Rotational Speed", color="green")

# Labels and Title
plt.xlabel("Time")
plt.ylabel("Sensor Readings")
plt.title("📈 Sensor Trends Over Time")
plt.legend()

# Show plot
plt.show()
