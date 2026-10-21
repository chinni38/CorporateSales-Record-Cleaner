"""
Corporate Sales Record Cleaner
Automated data sanitization pipeline using Pandas and NumPy.

Run:
    python sales_cleaner.py
Optional:
    python sales_cleaner.py --input raw_sales.csv --output cleaned_sales.csv
"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

def clean_sales(input_path: str, output_path: str):
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {input_file}. Put raw_sales.csv beside this script "
            "or pass the correct path using --input."
        )

    df = pd.read_csv(input_file)
    original_shape = df.shape
    original_nulls = df.isnull().sum()
    original_duplicates = int(df.duplicated().sum())
    baseline_summary = df.describe(include="all").transpose()
    baseline_summary.to_csv("baseline_summary_statistics.csv")

    # Remove fully identical rows (keep first occurrence).
    df = df.drop_duplicates().copy()

    # Standardize text fields: trim whitespace and lowercase consistently.
    text_columns = df.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        df[column] = df[column].astype("string").str.strip().str.lower()
        df[column] = df[column].replace({"": pd.NA, "nan": pd.NA})

    # Restore useful display casing for categorical columns.
    if "City" in df.columns:
        df["City"] = df["City"].str.title()
    if "Payment_Method" in df.columns:
        df["Payment_Method"] = df["Payment_Method"].str.title()

    # Parse dates consistently when present.
    if "Order_Date" in df.columns:
        df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # Missing-value strategy:
    # Numeric fields -> median (robust to outliers); categorical/text -> mode.
    for column in df.columns:
        if df[column].isna().any():
            if pd.api.types.is_numeric_dtype(df[column]):
                median_value = df[column].median()
                df[column] = df[column].fillna(median_value)
            else:
                modes = df[column].mode(dropna=True)
                replacement = modes.iloc[0] if not modes.empty else "Unknown"
                df[column] = df[column].fillna(replacement)

    # Ensure numeric columns are numeric and avoid invalid negative values.
    for column in ["Price", "Quantity", "Age", "Order_ID"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in ["Price", "Quantity", "Age"]:
        if column in df.columns:
            if df[column].isna().any():
                df[column] = df[column].fillna(df[column].median())
            df[column] = df[column].clip(lower=0)

    # Write cleaned output and final summary.
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    df.describe(include="all").transpose().to_csv("cleaned_summary_statistics.csv")

    report = [
        "CORPORATE SALES RECORD CLEANER — RUN REPORT",
        "=" * 48,
        f"Input file: {input_file}",
        f"Output file: {output_file}",
        f"Before shape (rows, columns): {original_shape}",
        f"After shape (rows, columns): {df.shape}",
        f"Exact duplicate rows removed: {original_duplicates}",
        "",
        "Missing values before cleaning:",
        original_nulls.to_string(),
        "",
        "Missing values after cleaning:",
        df.isnull().sum().to_string(),
        "",
        "Cleaning decisions:",
        "- Removed fully identical duplicate records.",
        "- Trimmed leading/trailing whitespace and standardized text casing.",
        "- Imputed numeric missing values with the median.",
        "- Imputed categorical/text missing values with the mode (or 'Unknown' if no mode exists).",
        "- Parsed Order_Date into YYYY-MM-DD format.",
        "- Clipped negative Price, Quantity, and Age values to zero.",
        "",
        "Baseline summary: baseline_summary_statistics.csv",
        "Cleaned summary: cleaned_summary_statistics.csv",
    ]
    Path("cleaning_report.txt").write_text("\n".join(report), encoding="utf-8")
    return df

def main():
    parser = argparse.ArgumentParser(description="Clean a raw corporate sales CSV dataset.")
    parser.add_argument("--input", default="raw_sales.csv", help="Path to raw input CSV")
    parser.add_argument("--output", default="cleaned_sales.csv", help="Path for cleaned output CSV")
    args = parser.parse_args()
    cleaned = clean_sales(args.input, args.output)
    print(f"Cleaning complete. Cleaned dataset shape: {cleaned.shape}")
    print("Generated: cleaned_sales.csv, baseline_summary_statistics.csv, "
          "cleaned_summary_statistics.csv, cleaning_report.txt")

if __name__ == "__main__":
    main()
