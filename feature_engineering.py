import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Create sample customer dataset
data = {
    "CustomerID": [101, 102, 103, 104, 105],
    "Name": ["Alice", "Bob", "Charlie", "David", "Emma"],
    "Age": [10, 25, 40, 65, 15],
    "Gender": ["Female", "Male", "Male", "Male", "Female"],
    "City": ["Bangalore", "Mysore", "Bangalore", "Mangalore", "Mysore"],
    "Income": [20000, 50000, 60000, 70000, 25000]
}

# Create DataFrame
df = pd.DataFrame(data)

# Save and reload the dataset (optional)
df.to_csv("customers.csv", index=False)
df = pd.read_csv("customers.csv")

# i. Create Age Group feature
def age_group(age):
    if age < 18:
        return "Child"
    elif age < 60:
        return "Adult"
    else:
        return "Senior"

df["Age_Group"] = df["Age"].apply(age_group)

# ii. One-Hot Encoding of categorical features
df = pd.get_dummies(df, columns=["Gender", "City", "Age_Group"])

# iii. Normalize numerical features using Min-Max Scaling
scaler = MinMaxScaler()

numerical_columns = ["Age", "Income"]

df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# iv. Save transformed data
df.to_csv("transformed_customers.csv", index=False)

print("Transformed Dataset:")
print(df)

print("\nTransformed data has been saved as 'transformed_customers.csv'")