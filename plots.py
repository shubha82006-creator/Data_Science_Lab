import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create sample data
data = {
    "Name": ["Alice", "Bob", "Charlie", "David", "Emma",
             "Frank", "Grace", "Henry", "Ivy", "Jack"],
    "Age": [21, 22, 20, 23, 21, 24, 22, 23, 20, 21],
    "Marks": [85, 90, 78, 88, 92, 75, 80, 95, 89, 84]
}

# Create DataFrame
df = pd.DataFrame(data)

# Save as CSV
df.to_csv("students.csv", index=False)

# Load CSV
df = pd.read_csv("students.csv")

# Create 'images' folder if it doesn't exist
os.makedirs("images", exist_ok=True)

# Histogram
plt.figure(figsize=(6,4))
sns.histplot(df["Marks"], bins=5, kde=True)
plt.title("Histogram of Marks")
plt.xlabel("Marks")
plt.ylabel("Frequency")
plt.savefig("images/histogram.png")
plt.show()

# Boxplot
plt.figure(figsize=(6,4))
sns.boxplot(y=df["Marks"])
plt.title("Boxplot of Marks")
plt.savefig("images/boxplot.png")
plt.show()