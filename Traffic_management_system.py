"""
Smart City Traffic Management System
Data Processing Pipeline for IoT Junction Sensor Data
"""

import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings("ignore")


# ------------------------------------------------------------------
# 1. LOAD MULTIPLE CSV FILES FROM DIFFERENT JUNCTIONS
# ------------------------------------------------------------------
def load_junction_data(folder_path="./traffic_data/"):
    """
    Loads all CSV files from a folder. Each file is assumed to be
    data from one junction, e.g. junction_1.csv, junction_2.csv ...
    """
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    dataframes = []

    if not all_files:
        print(f"No CSV files found in {folder_path}. Generating sample data instead.")
        return generate_sample_data()

    for file in all_files:
        try:
            df = pd.read_csv(file)
            junction_name = os.path.splitext(os.path.basename(file))[0]
            if "junction_id" not in df.columns:
                df["junction_id"] = junction_name
            dataframes.append(df)
            print(f"Loaded {file} -> {df.shape[0]} rows")
        except Exception as e:
            print(f"Error loading {file}: {e}")

    return dataframes


def generate_sample_data(n_junctions=3, n_records=200, seed=42):
    """
    Generates synthetic sample datasets simulating IoT sensor data
    (used when no real CSV files are available, for demo/testing).
    """
    np.random.seed(seed)
    dataframes = []
    weather_options = ["Clear", "Rain", "Fog", "Storm", "Cloudy"]

    base_time = datetime(2024, 1, 1, 0, 0)

    for j in range(1, n_junctions + 1):
        timestamps = [base_time + pd.Timedelta(minutes=15 * i) for i in range(n_records)]
        df = pd.DataFrame({
            "timestamp": timestamps,
            "junction_id": f"J{j}",
            "vehicle_count": np.random.randint(5, 300, n_records).astype(float),
            "avg_speed": np.random.normal(40, 12, n_records).round(1),
            "weather": np.random.choice(weather_options, n_records),
            "signal_timing_sec": np.random.choice([30, 45, 60, 90], n_records),
            "accident_flag": np.random.choice([0, 1], n_records, p=[0.95, 0.05]),
        })

        # Inject missing values
        for col in ["vehicle_count", "avg_speed", "weather", "signal_timing_sec"]:
            missing_idx = np.random.choice(df.index, size=int(0.08 * n_records), replace=False)
            df.loc[missing_idx, col] = np.nan

        # Inject corrupted / invalid records
        corrupt_idx = np.random.choice(df.index, size=5, replace=False)
        df.loc[corrupt_idx, "avg_speed"] = -999          # invalid negative speed
        corrupt_idx2 = np.random.choice(df.index, size=5, replace=False)
        df.loc[corrupt_idx2, "vehicle_count"] = 99999     # sensor spike/outlier

        # Inject duplicate timestamps (simulating overlapping sensor pings)
        dup_idx = np.random.choice(df.index[:-1], size=6, replace=False)
        for idx in dup_idx:
            df.loc[idx, "timestamp"] = df.loc[idx + 1, "timestamp"]

        dataframes.append(df)

    return dataframes


# ------------------------------------------------------------------
# 2. MERGE ALL DATASETS INTO A SINGLE TRAFFIC DATABASE
# ------------------------------------------------------------------
def merge_datasets(dataframes):
    merged_df = pd.concat(dataframes, ignore_index=True, sort=False)
    merged_df["timestamp"] = pd.to_datetime(merged_df["timestamp"], errors="coerce")
    merged_df.sort_values(["junction_id", "timestamp"], inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    print(f"\nMerged dataset shape: {merged_df.shape}")
    return merged_df


# ------------------------------------------------------------------
# 3. DETECT MISSING SENSOR READINGS AND CORRUPTED RECORDS
# ------------------------------------------------------------------
def detect_missing_and_corrupted(df):
    report = {}

    # Missing values per column
    missing_summary = df.isnull().sum()
    report["missing_values"] = missing_summary[missing_summary > 0]

    # Corrupted records: physically impossible values
    corrupted_mask = (
        (df["avg_speed"] < 0) |
        (df["avg_speed"] > 200) |
        (df["vehicle_count"] < 0) |
        (df["vehicle_count"] > 1000)
    )
    report["corrupted_count"] = int(corrupted_mask.sum())
    report["corrupted_records"] = df[corrupted_mask]

    # Mark corrupted values as NaN so they get handled by imputation
    df.loc[df["avg_speed"] < 0, "avg_speed"] = np.nan
    df.loc[df["avg_speed"] > 200, "avg_speed"] = np.nan
    df.loc[df["vehicle_count"] < 0, "vehicle_count"] = np.nan
    df.loc[df["vehicle_count"] > 1000, "vehicle_count"] = np.nan

    print("\n--- Missing Value Report ---")
    print(report["missing_values"])
    print(f"\nCorrupted records found & nulled out: {report['corrupted_count']}")

    return df, report


# ------------------------------------------------------------------
# 4. IDENTIFY DUPLICATE TIMESTAMPS FROM DIFFERENT SENSORS
# ------------------------------------------------------------------
def identify_duplicates(df):
    dup_mask = df.duplicated(subset=["junction_id", "timestamp"], keep=False)
    duplicates = df[dup_mask].sort_values(["junction_id", "timestamp"])

    print(f"\nDuplicate timestamp records found: {duplicates.shape[0]}")

    # Strategy: keep the record with the most complete data (least NaNs);
    # if tied, keep the first occurrence.
    df["_null_count"] = df.isnull().sum(axis=1)
    df.sort_values(["junction_id", "timestamp", "_null_count"], inplace=True)
    df_deduped = df.drop_duplicates(subset=["junction_id", "timestamp"], keep="first").copy()
    df_deduped.drop(columns="_null_count", inplace=True)
    df_deduped.reset_index(drop=True, inplace=True)

    print(f"Rows before dedup: {df.shape[0]}, after dedup: {df_deduped.shape[0]}")
    return df_deduped, duplicates


# ------------------------------------------------------------------
# 5. COMPARE MULTIPLE MISSING VALUE IMPUTATION TECHNIQUES
# ------------------------------------------------------------------
def compare_imputation_techniques(df):
    numeric_cols = ["vehicle_count", "avg_speed", "signal_timing_sec"]
    results = {}

    original = df[numeric_cols].copy()

    # --- Technique 1: Mean Imputation ---
    mean_imputer = SimpleImputer(strategy="mean")
    mean_imputed = pd.DataFrame(
        mean_imputer.fit_transform(original), columns=numeric_cols
    )

    # --- Technique 2: Median Imputation ---
    median_imputer = SimpleImputer(strategy="median")
    median_imputed = pd.DataFrame(
        median_imputer.fit_transform(original), columns=numeric_cols
    )

    # --- Technique 3: Forward Fill / Backward Fill (time-series aware) ---
    ffill_imputed = original.ffill().bfill()
    # --- Technique 4: KNN Imputation ---
    knn_imputer = KNNImputer(n_neighbors=5)
    knn_imputed = pd.DataFrame(
        knn_imputer.fit_transform(original), columns=numeric_cols
    )

    # --- Technique 5: Interpolation (linear, time-based) ---
    interp_imputed = original.interpolate(method="linear", limit_direction="both")

    # Compare using standard deviation preservation as a simple quality metric
    for name, imputed_df in [
        ("Mean", mean_imputed), ("Median", median_imputed),
        ("Forward/Backward Fill", ffill_imputed),
        ("KNN", knn_imputed), ("Linear Interpolation", interp_imputed)
    ]:
        std_diff = (imputed_df.std() - original.std()).abs().mean()
        results[name] = {"mean_std_deviation_shift": round(std_diff, 3)}

    comparison_df = pd.DataFrame(results).T
    print("\n--- Imputation Technique Comparison (lower shift = better) ---")
    print(comparison_df)

    # Choose best technique: lowest std deviation shift
    best_method = comparison_df["mean_std_deviation_shift"].idxmin()
    print(f"\nSelected imputation technique: {best_method}")

    final_imputed = {
        "Mean": mean_imputed, "Median": median_imputed,
        "Forward/Backward Fill": ffill_imputed,
        "KNN": knn_imputed, "Linear Interpolation": interp_imputed
    }[best_method]

    df[numeric_cols] = final_imputed

    # Categorical column: impute with mode
    if "weather" in df.columns:
        df["weather"] = df["weather"].fillna(df["weather"].mode()[0])

    return df, comparison_df, best_method


# ------------------------------------------------------------------
# 6. GENERATE TRAFFIC DENSITY REPORTS BY LOCATION
# ------------------------------------------------------------------
def classify_congestion(vehicle_count):
    if vehicle_count < 50:
        return "Low"
    elif vehicle_count < 150:
        return "Moderate"
    elif vehicle_count < 250:
        return "High"
    else:
        return "Severe"


def generate_density_reports(df):
    df["congestion_level"] = df["vehicle_count"].apply(classify_congestion)

    report = df.groupby("junction_id").agg(
        avg_vehicle_count=("vehicle_count", "mean"),
        max_vehicle_count=("vehicle_count", "max"),
        avg_speed=("avg_speed", "mean"),
        accident_count=("accident_flag", "sum"),
        total_records=("vehicle_count", "count")
    ).round(2)

    congestion_dist = pd.crosstab(df["junction_id"], df["congestion_level"])

    report = report.join(congestion_dist)
    report["primary_congestion_level"] = df.groupby("junction_id")["congestion_level"] \
        .agg(lambda x: x.value_counts().idxmax())

    print("\n--- Traffic Density Report by Junction ---")
    print(report)

    return df, report


# ------------------------------------------------------------------
# 7. SAVE THE PROCESSED DATASET
# ------------------------------------------------------------------
def save_outputs(df, report, output_dir="./traffic_output/"):
    os.makedirs(output_dir, exist_ok=True)

    processed_path = os.path.join(output_dir, "processed_traffic_data.csv")
    report_path = os.path.join(output_dir, "traffic_density_report.csv")

    df.to_csv(processed_path, index=False)
    report.to_csv(report_path)

    print(f"\nProcessed dataset saved to: {processed_path}")
    print(f"Density report saved to: {report_path}")


# ------------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------------
def main():
    print("=" * 60)
    print("SMART CITY TRAFFIC MANAGEMENT - DATA PIPELINE")
    print("=" * 60)

    # Step 1: Load
    dataframes = load_junction_data(folder_path="./traffic_data/")

    # Step 2: Merge
    merged_df = merge_datasets(dataframes)

    # Step 3: Detect missing/corrupted
    cleaned_df, missing_report = detect_missing_and_corrupted(merged_df)

    # Step 4: Deduplicate
    deduped_df, duplicate_records = identify_duplicates(cleaned_df)

    # Step 5: Impute
    imputed_df, imputation_comparison, best_method = compare_imputation_techniques(deduped_df)

    # Step 6: Density report
    final_df, density_report = generate_density_reports(imputed_df)

    # Step 7: Save
    save_outputs(final_df, density_report)

    print("\nPipeline completed successfully.")
    return final_df, density_report


if __name__ == "__main__":
    final_data, density_report = main()