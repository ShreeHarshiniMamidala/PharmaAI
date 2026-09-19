from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(
    r"C:\Users\mamid\OneDrive\Desktop\PharmaAI"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling"
    / "2025_Q1"
    / "drug-event-0001-of-0028.parquet"
)

print("=" * 70)
print("PharmaAI - Extracted Dataset Validation")
print("=" * 70)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_parquet(MODEL_FILE)

print(f"\nDataset: {MODEL_FILE}")
print(f"\nRows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("COLUMNS")
print("=" * 70)

for i, column in enumerate(df.columns, start=1):
    print(f"{i:2}. {column}")


# ============================================================
# FIRST 5 ROWS
# ============================================================

print("\n" + "=" * 70)
print("FIRST 5 ROWS")
print("=" * 70)

print(df.head().to_string())


# ============================================================
# DATA TYPES
# ============================================================

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(df.dtypes)


# ============================================================
# MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_percent": (
        df.isna().mean() * 100
    ).round(2)
})

missing = missing.sort_values(
    "missing_percent",
    ascending=False
)

print(missing.to_string())


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TARGET: serious")
print("=" * 70)

print("\nCounts:")
print(
    df["serious"]
    .value_counts(dropna=False)
)

print("\nPercentages:")

print(
    (
        df["serious"]
        .value_counts(
            normalize=True,
            dropna=False
        )
        * 100
    ).round(2)
)


# ============================================================
# AGE VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("PATIENT AGE")
print("=" * 70)

print(
    df["patient_age_years"]
    .describe()
)

invalid_age = df[
    (df["patient_age_years"] < 0)
    |
    (df["patient_age_years"] > 120)
]

print(
    f"\nPotentially invalid ages: "
    f"{len(invalid_age):,}"
)


# ============================================================
# WEIGHT VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("PATIENT WEIGHT")
print("=" * 70)

print(
    df["patient_weight"]
    .describe()
)


# ============================================================
# SEX DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("PATIENT SEX")
print("=" * 70)

print(
    df["patient_sex"]
    .value_counts(
        dropna=False
    )
)


# ============================================================
# REPORTER QUALIFICATION
# ============================================================

print("\n" + "=" * 70)
print("REPORTER QUALIFICATION")
print("=" * 70)

print(
    df["reporter_qualification"]
    .value_counts(
        dropna=False
    )
)


# ============================================================
# DRUG COUNTS
# ============================================================

print("\n" + "=" * 70)
print("DRUG COUNTS")
print("=" * 70)

drug_columns = [
    "num_drugs",
    "num_suspect_drugs",
    "num_concomitant_drugs",
    "num_interacting_drugs"
]

print(
    df[drug_columns]
    .describe()
    .to_string()
)


# ============================================================
# REACTION COUNTS
# ============================================================

print("\n" + "=" * 70)
print("REACTION COUNTS")
print("=" * 70)

print(
    df["num_reactions"]
    .describe()
)


# ============================================================
# DUPLICATE REPORT IDS
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE SAFETY REPORT IDs")
print("=" * 70)

duplicate_ids = df[
    "safetyreportid"
].duplicated().sum()

print(
    f"Duplicate safetyreportid values: "
    f"{duplicate_ids:,}"
)


# ============================================================
# UNIQUE VALUES
# ============================================================

print("\n" + "=" * 70)
print("CARDINALITY")
print("=" * 70)

categorical_columns = [
    "patient_sex",
    "country",
    "report_type",
    "reporter_qualification",
    "drug_indications",
    "drug_routes",
    "drug_dosage_forms"
]

for column in categorical_columns:

    print(
        f"{column:25} : "
        f"{df[column].nunique(dropna=True):,}"
    )


# ============================================================
# LEAKAGE CHECK
# ============================================================

print("\n" + "=" * 70)
print("TARGET LEAKAGE CHECK")
print("=" * 70)

forbidden_columns = [
    "seriousnessdeath",
    "seriousnesshospitalization",
    "seriousnesslifethreatening",
    "seriousnessdisabling",
    "seriousnesscongenital",
    "seriousnessother",
    "reaction_outcomes"
]

found = [
    column
    for column in forbidden_columns
    if column in df.columns
]

if not found:

    print(
        "PASS: No obvious seriousness/outcome "
        "columns are present in the modeling dataset."
    )

else:

    print("WARNING: Potential leakage columns found:")

    for column in found:
        print(f" - {column}")


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)