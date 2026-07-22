# ==========================================
# Experiment 1
# University Admission Analytics System
# ==========================================

import pandas as pd
import numpy as np

# ==========================================
# 1. Import Dataset
# ==========================================

# Replace with your dataset name
df = pd.read_csv("Question-5/University_admission.csv")

print("=" * 60)
print("FIRST 5 RECORDS")
print("=" * 60)
print(df.head())

print("\nDataset Shape:", df.shape)

# ==========================================
# 2. Dataset Profiling Report
# ==========================================

def profiling_report(data):

    print("\n" + "="*60)
    print("DATASET PROFILING REPORT")
    print("="*60)

    print("\nData Types")
    print(data.dtypes)

    print("\nMissing Values")
    print(data.isnull().sum())

    print("\nDuplicate Rows")
    print(data.duplicated().sum())

    print("\nMemory Usage")

    memory = data.memory_usage(deep=True).sum()/1024

    print(f"{memory:.2f} KB")

    print("\nUnique Values")

    print(data.nunique())

profiling_report(df)

# ==========================================
# 3. Classify Attributes
# ==========================================

def classify_columns(data):

    numerical = []
    categorical = []
    ordinal = []

    ordinal_keywords = [
        "grade",
        "rank",
        "year",
        "semester",
        "category",
        "education"
    ]

    for column in data.columns:

        if pd.api.types.is_numeric_dtype(data[column]):

            numerical.append(column)

        else:

            found = False

            for word in ordinal_keywords:

                if word.lower() in column.lower():

                    ordinal.append(column)
                    found = True
                    break

            if not found:
                categorical.append(column)

    return numerical, categorical, ordinal

num_cols, cat_cols, ord_cols = classify_columns(df)

print("\nNumerical Columns")
print(num_cols)

print("\nCategorical Columns")
print(cat_cols)

print("\nOrdinal Columns")
print(ord_cols)

# ==========================================
# 4. Identify Inconsistencies
# ==========================================

print("\n" + "="*60)
print("DATA INCONSISTENCIES")
print("="*60)

# Duplicate Application Numbers
if "Application_No" in df.columns:

    duplicates = df[df.duplicated("Application_No", keep=False)]

    print("\nDuplicate Application Numbers:")

    print(duplicates)

# Invalid Entrance Marks

if "Entrance_Score" in df.columns:

    invalid_marks = df[
        (df["Entrance_Score"] < 0) |
        (df["Entrance_Score"] > 100)
    ]

    print("\nInvalid Entrance Scores")

    print(invalid_marks)

# Invalid Board Percentage

if "Board_Percentage" in df.columns:

    invalid_board = df[
        (df["Board_Percentage"] < 0) |
        (df["Board_Percentage"] > 100)
    ]

    print("\nInvalid Board Percentage")

    print(invalid_board)

# Invalid Age

if "Age" in df.columns:

    invalid_age = df[
        (df["Age"] < 16) |
        (df["Age"] > 30)
    ]

    print("\nInvalid Age")

    print(invalid_age)

# ==========================================
# 5. Admission Statistics Report
# ==========================================

def admission_statistics(data):

    print("\n" + "="*60)
    print("ADMISSION STATISTICS")
    print("="*60)

    print("\nTotal Students:", len(data))

    if "Admission_Status" in data.columns:

        print("\nAdmission Status")

        print(data["Admission_Status"].value_counts())

    if "Branch" in data.columns:

        print("\nBranch Preferences")

        print(data["Branch"].value_counts())

    if "Reservation_Category" in data.columns:

        print("\nReservation Categories")

        print(data["Reservation_Category"].value_counts())

    if "Family_Income" in data.columns:

        print("\nAverage Family Income")

        print(data["Family_Income"].mean())

    if "Entrance_Score" in data.columns:

        print("\nAverage Entrance Score")

        print(data["Entrance_Score"].mean())

    if "Board_Percentage" in data.columns:

        print("\nAverage Board Percentage")

        print(data["Board_Percentage"].mean())

admission_statistics(df)

# ==========================================
# 6. Data Cleaning
# ==========================================

clean_df = df.copy()

# Remove duplicate rows
clean_df.drop_duplicates(inplace=True)

# Fill missing numerical values

for col in clean_df.select_dtypes(include=np.number):

    clean_df[col].fillna(clean_df[col].median(), inplace=True)

# Fill missing categorical values

for col in clean_df.select_dtypes(include="object"):

    clean_df[col].fillna(clean_df[col].mode()[0], inplace=True)

# Remove invalid ages

if "Age" in clean_df.columns:

    clean_df = clean_df[
        (clean_df["Age"] >= 16) &
        (clean_df["Age"] <= 30)
    ]

# Remove invalid entrance marks

if "Entrance_Score" in clean_df.columns:

    clean_df = clean_df[
        (clean_df["Entrance_Score"] >= 0) &
        (clean_df["Entrance_Score"] <= 100)
    ]

# Remove invalid board percentages

if "Board_Percentage" in clean_df.columns:

    clean_df = clean_df[
        (clean_df["Board_Percentage"] >= 0) &
        (clean_df["Board_Percentage"] <= 100)
    ]

# ==========================================
# Export Clean Dataset
# ==========================================

clean_df.to_csv("cleaned_university_admission.csv", index=False)

print("\n" + "="*60)
print("Cleaned dataset exported successfully!")
print("File Name : cleaned_university_admission.csv")
print("="*60)