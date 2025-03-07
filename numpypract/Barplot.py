import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("ai4i2020.csv")

# Sum the occurrences of each failure type
failure_types = ["TWF", "HDF", "PWF", "OSF", "RNF"]
failure_counts = df[failure_types].sum()

# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(
    failure_counts, labels=failure_counts.index, autopct='%1.1f%%',
    colors=["blue", "red", "green", "purple", "orange"], startangle=140
)

# Title
plt.title("⚙️ Contribution of Different Failure Types in Overall Data")

# Show plot
plt.show()
