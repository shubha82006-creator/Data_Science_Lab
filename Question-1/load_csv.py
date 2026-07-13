import pandas as pd

# Step 1: Create sample data
data = {
    "Name": ["Alice", "Bob", "Charlie", "David", "Emma",
             "Frank", "Grace", "Henry", "Ivy", "Jack"],
    "Age": [21, 22, 20, 23, 21, 24, 22, 23, 20, 21],
    "Marks": [85, 90, 78, 88, 92, 75, 80, 95, 89, 84]
}

# Step 2: Convert the data into a DataFrame
sample_df = pd.DataFrame(data)

# Step 3: Save the DataFrame as a CSV file
sample_df.to_csv("students.csv", index=False)

# Step 4: Load the CSV file into a DataFrame
df = pd.read_csv("students.csv")

# Step 5: Display the first 10 rows
print("First 10 Rows:")
print(df.head(10))

# Step 6: Show the number of rows and columns
print("\nNumber of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])

# Step 7: Display summary statistics for numerical columns
print("\nSummary Statistics:")
print(df.describe())