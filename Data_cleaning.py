# ==============================================
# Advanced Data Cleaning & Feature Engineering
# ==============================================

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv(
    r"C:\Users\flipkart\OneDrive\Desktop\Data Science\Question-7\ecommerce.csv"
)

print("="*60)
print("ORIGINAL DATASET")
print("="*60)
print(df.head())

# =====================================================
# 1. Detect Data Quality Issues
# =====================================================

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Customer IDs")
print(df[df.duplicated(subset="Customer_ID")])

print("\nDate Formats")
print(df["Purchase_Date"].head())

print("\nCurrencies")
print(df["Currency"].unique())

print("\nIncorrect Age Values")
print(df[(df["Age"] < 18) | (df["Age"] > 80)])

print("\nTypographical Errors")
print(df["City"].unique())

# Detect Outliers using IQR

Q1 = df["Salary"].quantile(0.25)
Q3 = df["Salary"].quantile(0.75)

IQR = Q3 - Q1

outliers = df[
    (df["Salary"] < Q1 - 1.5 * IQR) |
    (df["Salary"] > Q3 + 1.5 * IQR)
]

print("\nOutliers")
print(outliers)

# =====================================================
# 2. Why Data Quality Issues Affect Analytics
# =====================================================

print("""
Missing Values -> Reduce model accuracy
Duplicate IDs -> Duplicate customer records
Different Date Formats -> Incorrect time analysis
Currency Mismatch -> Wrong financial calculations
Incorrect Age -> Misleading statistics
Outliers -> Distort machine learning models
Typographical Errors -> Duplicate categories
""")

# =====================================================
# 3. Missing Value Handling
# =====================================================

# Mean Imputation
mean_df = df.copy()
mean_df["Salary"] = mean_df["Salary"].fillna(mean_df["Salary"].mean())

# Median Imputation
median_df = df.copy()
median_df["Salary"] = median_df["Salary"].fillna(median_df["Salary"].median())

# Mode Imputation
mode_df = df.copy()
mode_df["City"] = mode_df["City"].fillna(mode_df["City"].mode()[0])

# Forward Fill
ffill_df = df.copy()
ffill_df = ffill_df.fillna(method="ffill")

# Backward Fill
bfill_df = df.copy()
bfill_df = bfill_df.fillna(method="bfill")

print("\nComparison of Imputation")

comparison = pd.DataFrame({

    "Original": df["Salary"],
    "Mean": mean_df["Salary"],
    "Median": median_df["Salary"],
    "ForwardFill": ffill_df["Salary"],
    "BackwardFill": bfill_df["Salary"]

})

print(comparison.head())

# =====================================================
# 4. Cleaning Dataset
# =====================================================

# Remove Duplicate IDs

df = df.drop_duplicates(subset="Customer_ID")

# Convert Date

df["Purchase_Date"] = pd.to_datetime(df["Purchase_Date"])

# Currency Conversion Example

usd_rate = 83

df.loc[df["Currency"] == "USD", "Salary"] = (
    df.loc[df["Currency"] == "USD", "Salary"] * usd_rate
)

df["Currency"] = "INR"

# Correct Invalid Ages

df = df[(df["Age"] >= 18) & (df["Age"] <= 80)]

# Remove Outliers

Q1 = df["Salary"].quantile(0.25)
Q3 = df["Salary"].quantile(0.75)

IQR = Q3 - Q1

df = df[
    (df["Salary"] >= Q1 - 1.5 * IQR) &
    (df["Salary"] <= Q3 + 1.5 * IQR)
]

# Correct Typographical Errors

df["City"] = df["City"].replace({

    "Banglore": "Bangalore",
    "Bangaluru": "Bangalore",
    "Delhii": "Delhi",
    "Mumbia": "Mumbai"

})

# Fill Missing Salary using Mean

df["Salary"] = df["Salary"].fillna(df["Salary"].mean())

# =====================================================
# 5. Standardization
# =====================================================

standard = StandardScaler()

df["Salary_Standardized"] = standard.fit_transform(df[["Salary"]])

# =====================================================
# 6. Normalization
# =====================================================

normalize = MinMaxScaler()

df["Salary_Normalized"] = normalize.fit_transform(df[["Salary"]])

# =====================================================
# 7. Label Encoding
# =====================================================

if "Gender" in df.columns:

    encoder = LabelEncoder()

    df["Gender"] = encoder.fit_transform(df["Gender"])

# =====================================================
# 8. One Hot Encoding
# =====================================================

df = pd.get_dummies(df, columns=["City"])

# =====================================================
# 9. Feature Engineering
# =====================================================

# Annual Income

df["Annual_Income"] = df["Salary"] * 12

# Age Group

df["Age_Group"] = pd.cut(

    df["Age"],

    bins=[18,30,45,60,100],

    labels=["Young","Adult","Middle Age","Senior"]

)

# Spending Category

df["Spending_Category"] = pd.cut(

    df["Spending"],

    bins=[0,5000,15000,50000],

    labels=["Low","Medium","High"]

)

# Customer Value Index

df["Customer_Value_Index"] = (

    df["Annual_Income"] *

    df["Purchase_Frequency"]

) / 1000

# =====================================================
# 10. Compare Before and After
# =====================================================

print("\nShape Before Cleaning:", mean_df.shape)
print("Shape After Cleaning :", df.shape)

print("\nCleaned Dataset")

print(df.head())

print("\nDataset Information")

print(df.info())

# =====================================================
# Save Cleaned Dataset
# =====================================================

df.to_csv("ecommerce_cleaned.csv", index=False)

print("\nPreprocessing Completed Successfully.")
print("Cleaned dataset saved as ecommerce_cleaned.csv")