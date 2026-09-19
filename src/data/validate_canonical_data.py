from pathlib import Path
from collections import Counter
import gc
import sys

import pandas as pd
import pyarrow.parquet as pq


# ============================================================
# PHARMAAI - CANONICAL FAERS DATA VALIDATION
# ============================================================
#
# PURPOSE
# -------
# Validate the canonical report × suspect-drug × adverse-event
# datasets created by extract_faers.py.
#
# DATASETS
# --------
# 2025 Q1-Q4 -> Development
# 2026 Q1    -> Temporal test
#
# IMPORTANT
# ---------
# The files contain millions of rows.
#
# Therefore:
#   - PyArrow reads the Parquet files in batches.
#   - We do NOT load the entire dataset into pandas.
#
# ============================================================


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\mamid\OneDrive\Desktop\PharmaAI"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "validation"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. INPUT DATASETS
# ============================================================

DATASETS = {

    "2025 DEVELOPMENT": (
        PROCESSED_DIR
        / "faers_2025_clean.parquet"
    ),

    "2026 Q1 TEMPORAL TEST": (
        PROCESSED_DIR
        / "faers_2026_Q1_clean.parquet"
    ),
}


# ============================================================
# 3. EXPECTED ROW COUNTS
# ============================================================
#
# These come from the extraction output.
#
# Validation will flag a mismatch.
# ============================================================

EXPECTED_ROWS = {

    "2025 DEVELOPMENT":
        16_296_363,

    "2026 Q1 TEMPORAL TEST":
        4_420_077,
}


# ============================================================
# 4. EXPECTED COLUMNS
# ============================================================

EXPECTED_COLUMNS = [

    "safetyreportid",

    "year",
    "quarter",

    "receivedate",
    "receiptdate",

    "suspect_drug",
    "adverse_event",

    "active_substance",
    "generic_name",

    "drug_indication",
    "drug_route",
    "drug_dosage_form",

    "serious",

    "patient_age_years",
    "patient_sex",

    "country",

    "report_type",
    "reporter_qualification",

    "num_drugs",
    "num_reactions",
]


# ============================================================
# 5. CRITICAL COLUMNS
# ============================================================
#
# These must NOT be missing because they define the
# drug-event observation.
# ============================================================

CRITICAL_COLUMNS = [

    "safetyreportid",
    "suspect_drug",
    "adverse_event",
]


# ============================================================
# 6. BATCH SIZE
# ============================================================
#
# 250,000 rows gives a good balance between speed and memory.
#
# Reduce to 100,000 if your computer has limited RAM.
# ============================================================

BATCH_SIZE = 250_000


# ============================================================
# 7. HELPER FUNCTIONS
# ============================================================

def print_header(title):

    print("\n")
    print("=" * 78)
    print(title)
    print("=" * 78)


def print_subheader(title):

    print("\n")
    print("-" * 78)
    print(title)
    print("-" * 78)


def safe_percent(
    numerator,
    denominator
):

    if denominator == 0:
        return 0.0

    return (
        numerator
        / denominator
        * 100
    )


# ============================================================
# 8. BASIC FILE VALIDATION
# ============================================================

def validate_file_exists(
    dataset_name,
    file_path
):

    print_subheader(
        "FILE CHECK"
    )

    print(
        f"Dataset : {dataset_name}"
    )

    print(
        f"Path    : {file_path}"
    )


    if not file_path.exists():

        print(
            "\n❌ FILE NOT FOUND"
        )

        return False


    size_mb = (
        file_path.stat().st_size
        / (1024 ** 2)
    )


    print(
        f"Size    : {size_mb:,.2f} MB"
    )

    print(
        "\n✅ File exists."
    )


    return True


# ============================================================
# 9. SCHEMA VALIDATION
# ============================================================

def validate_schema(
    parquet_file
):

    print_subheader(
        "SCHEMA CHECK"
    )


    actual_columns = (
        parquet_file.schema.names
    )


    print(
        f"Columns found: "
        f"{len(actual_columns)}"
    )


    for index, column in enumerate(
        actual_columns,
        start=1
    ):

        print(
            f"{index:>2}. {column}"
        )


    missing_columns = [

        column

        for column
        in EXPECTED_COLUMNS

        if column not in actual_columns
    ]


    unexpected_columns = [

        column

        for column
        in actual_columns

        if column not in EXPECTED_COLUMNS
    ]


    if missing_columns:

        print(
            "\n❌ Missing expected columns:"
        )

        for column in missing_columns:

            print(
                f"   - {column}"
            )

    else:

        print(
            "\n✅ All expected columns found."
        )


    if unexpected_columns:

        print(
            "\nℹ️ Additional columns:"
        )

        for column in unexpected_columns:

            print(
                f"   - {column}"
            )


    return (
        missing_columns,
        unexpected_columns
    )


# ============================================================
# 10. PARQUET METADATA
# ============================================================

def validate_metadata(
    dataset_name,
    parquet_file
):

    print_subheader(
        "PARQUET METADATA"
    )


    metadata = (
        parquet_file.metadata
    )


    row_count = (
        metadata.num_rows
    )


    row_groups = (
        metadata.num_row_groups
    )


    print(
        f"Rows       : "
        f"{row_count:,}"
    )

    print(
        f"Row groups : "
        f"{row_groups:,}"
    )


    expected = EXPECTED_ROWS.get(
        dataset_name
    )


    if expected is not None:

        print(
            f"Expected   : "
            f"{expected:,}"
        )


        if row_count == expected:

            print(
                "\n✅ Row count matches "
                "extraction output."
            )

        else:

            print(
                "\n❌ ROW COUNT MISMATCH"
            )

            print(
                f"Difference: "
                f"{row_count - expected:+,}"
            )


    return row_count


# ============================================================
# 11. STREAMING DATA VALIDATION
# ============================================================

def stream_validate(
    dataset_name,
    file_path,
    available_columns
):

    print_subheader(
        "STREAMING DATA VALIDATION"
    )


    # ========================================================
    # COUNTERS
    # ========================================================

    rows_processed = 0


    # Critical missingness

    missing_critical = {

        column: 0

        for column
        in CRITICAL_COLUMNS
    }


    # --------------------------------------------------------
    # Unique values
    # --------------------------------------------------------

    unique_reports = set()

    unique_drugs = set()

    unique_events = set()

    unique_pairs = set()


    # --------------------------------------------------------
    # Duplicate report-drug-event key
    #
    # This can consume memory because there can be millions
    # of keys. We store a compact tuple.
    # --------------------------------------------------------

    seen_observations = set()

    duplicate_observations = 0


    # --------------------------------------------------------
    # Distribution counters
    # --------------------------------------------------------

    year_counter = Counter()

    quarter_counter = Counter()

    serious_counter = Counter()

    sex_counter = Counter()

    country_counter = Counter()


    # --------------------------------------------------------
    # Numeric statistics
    # --------------------------------------------------------

    age_count = 0

    age_missing = 0

    age_sum = 0.0

    age_min = None

    age_max = None

    invalid_age_count = 0


    max_num_drugs = 0

    max_num_reactions = 0


    num_drugs_sum = 0

    num_reactions_sum = 0

    numeric_count = 0


    # --------------------------------------------------------
    # Suspicious counts
    # --------------------------------------------------------

    zero_drug_count = 0

    zero_reaction_count = 0

    very_high_drug_count = 0

    very_high_reaction_count = 0


    # ========================================================
    # ONLY READ COLUMNS WE NEED
    # ========================================================

    wanted_columns = [

        "safetyreportid",

        "year",
        "quarter",

        "suspect_drug",
        "adverse_event",

        "serious",

        "patient_age_years",
        "patient_sex",

        "country",

        "num_drugs",
        "num_reactions",
    ]


    columns_to_read = [

        column

        for column
        in wanted_columns

        if column in available_columns
    ]


    parquet_file = pq.ParquetFile(
        file_path
    )


    batch_number = 0


    # ========================================================
    # BATCH LOOP
    # ========================================================

    for batch in parquet_file.iter_batches(

        batch_size=BATCH_SIZE,

        columns=columns_to_read
    ):

        batch_number += 1


        df = batch.to_pandas()


        batch_rows = len(
            df
        )


        rows_processed += (
            batch_rows
        )


        print(
            f"\rBatch {batch_number:,} | "
            f"Rows processed: "
            f"{rows_processed:,}",
            end="",
            flush=True
        )


        # ====================================================
        # CRITICAL MISSINGNESS
        # ====================================================

        for column in CRITICAL_COLUMNS:

            if column not in df.columns:
                continue


            missing_mask = (

                df[column].isna()

                |

                (
                    df[column]
                    .astype(str)
                    .str.strip()
                    == ""
                )
            )


            missing_critical[
                column
            ] += int(
                missing_mask.sum()
            )


        # ====================================================
        # UNIQUE COUNTS + DUPLICATES
        # ====================================================

        required = {
            "safetyreportid",
            "suspect_drug",
            "adverse_event"
        }


        if required.issubset(
            df.columns
        ):


            # ------------------------------------------------
            # Drop missing critical values for key operations
            # ------------------------------------------------

            key_df = df[
                [
                    "safetyreportid",
                    "suspect_drug",
                    "adverse_event"
                ]
            ].dropna()


            # ------------------------------------------------
            # Iterate only through key columns
            # ------------------------------------------------

            for report_id, drug, event in (
                key_df.itertuples(
                    index=False,
                    name=None
                )
            ):

                report_id = str(
                    report_id
                )

                drug = str(
                    drug
                )

                event = str(
                    event
                )


                unique_reports.add(
                    report_id
                )

                unique_drugs.add(
                    drug
                )

                unique_events.add(
                    event
                )


                pair = (
                    drug,
                    event
                )

                unique_pairs.add(
                    pair
                )


                observation_key = (
                    report_id,
                    drug,
                    event
                )


                if (
                    observation_key
                    in seen_observations
                ):

                    duplicate_observations += 1

                else:

                    seen_observations.add(
                        observation_key
                    )


        # ====================================================
        # YEAR
        # ====================================================

        if "year" in df.columns:

            values = (
                df["year"]
                .dropna()
                .value_counts()
            )


            for key, value in values.items():

                year_counter[
                    str(key)
                ] += int(
                    value
                )


        # ====================================================
        # QUARTER
        # ====================================================

        if "quarter" in df.columns:

            values = (
                df["quarter"]
                .fillna("MISSING")
                .value_counts()
            )


            for key, value in values.items():

                quarter_counter[
                    str(key)
                ] += int(
                    value
                )


        # ====================================================
        # SERIOUS
        # ====================================================

        if "serious" in df.columns:

            values = (
                df["serious"]
                .fillna("MISSING")
                .astype(str)
                .value_counts()
            )


            for key, value in values.items():

                serious_counter[
                    key
                ] += int(
                    value
                )


        # ====================================================
        # PATIENT SEX
        # ====================================================

        if "patient_sex" in df.columns:

            values = (
                df["patient_sex"]
                .fillna("MISSING")
                .astype(str)
                .value_counts()
            )


            for key, value in values.items():

                sex_counter[
                    key
                ] += int(
                    value
                )


        # ====================================================
        # COUNTRY
        # ====================================================

        if "country" in df.columns:

            values = (
                df["country"]
                .fillna("MISSING")
                .astype(str)
                .value_counts()
            )


            for key, value in values.items():

                country_counter[
                    key
                ] += int(
                    value
                )


        # ====================================================
        # AGE
        # ====================================================

        if "patient_age_years" in df.columns:

            ages = pd.to_numeric(
                df[
                    "patient_age_years"
                ],
                errors="coerce"
            )


            age_missing += int(
                ages.isna().sum()
            )


            valid_ages = (
                ages.dropna()
            )


            if not valid_ages.empty:

                age_count += len(
                    valid_ages
                )


                age_sum += float(
                    valid_ages.sum()
                )


                current_min = float(
                    valid_ages.min()
                )

                current_max = float(
                    valid_ages.max()
                )


                if (
                    age_min is None
                    or
                    current_min < age_min
                ):

                    age_min = current_min


                if (
                    age_max is None
                    or
                    current_max > age_max
                ):

                    age_max = current_max


                invalid_age_count += int(
                    (
                        (valid_ages < 0)
                        |
                        (valid_ages > 120)
                    ).sum()
                )


        # ====================================================
        # NUM DRUGS
        # ====================================================

        if "num_drugs" in df.columns:

            num_drugs = pd.to_numeric(
                df["num_drugs"],
                errors="coerce"
            )


            valid = (
                num_drugs.dropna()
            )


            if not valid.empty:

                current_max = int(
                    valid.max()
                )


                max_num_drugs = max(
                    max_num_drugs,
                    current_max
                )


                num_drugs_sum += float(
                    valid.sum()
                )


                zero_drug_count += int(
                    (valid <= 0).sum()
                )


                very_high_drug_count += int(
                    (valid > 100).sum()
                )


        # ====================================================
        # NUM REACTIONS
        # ====================================================

        if "num_reactions" in df.columns:

            num_reactions = pd.to_numeric(
                df["num_reactions"],
                errors="coerce"
            )


            valid = (
                num_reactions.dropna()
            )


            if not valid.empty:

                current_max = int(
                    valid.max()
                )


                max_num_reactions = max(
                    max_num_reactions,
                    current_max
                )


                num_reactions_sum += float(
                    valid.sum()
                )


                zero_reaction_count += int(
                    (valid <= 0).sum()
                )


                very_high_reaction_count += int(
                    (valid > 50).sum()
                )


        numeric_count += (
            batch_rows
        )


        # ====================================================
        # CLEAN MEMORY
        # ====================================================

        del df
        del batch

        gc.collect()


    print()


    # ========================================================
    # RESULTS
    # ========================================================

    results = {

        "dataset":
            dataset_name,

        "rows":
            rows_processed,

        "unique_reports":
            len(unique_reports),

        "unique_suspect_drugs":
            len(unique_drugs),

        "unique_adverse_events":
            len(unique_events),

        "unique_drug_event_pairs":
            len(unique_pairs),

        "duplicate_report_drug_event_rows":
            duplicate_observations,

        "missing_safetyreportid":
            missing_critical[
                "safetyreportid"
            ],

        "missing_suspect_drug":
            missing_critical[
                "suspect_drug"
            ],

        "missing_adverse_event":
            missing_critical[
                "adverse_event"
            ],

        "age_missing":
            age_missing,

        "age_invalid":
            invalid_age_count,

        "age_min":
            age_min,

        "age_max":
            age_max,

        "age_mean":
            (
                age_sum / age_count
                if age_count > 0
                else None
            ),

        "max_num_drugs":
            max_num_drugs,

        "max_num_reactions":
            max_num_reactions,

        "zero_num_drugs":
            zero_drug_count,

        "zero_num_reactions":
            zero_reaction_count,

        "rows_num_drugs_gt_100":
            very_high_drug_count,

        "rows_num_reactions_gt_50":
            very_high_reaction_count,
    }


    # ========================================================
    # PRINT VALIDATION RESULTS
    # ========================================================

    print_subheader(
        "CORE DATASET STATISTICS"
    )


    print(
        f"Rows                    : "
        f"{rows_processed:,}"
    )

    print(
        f"Unique safety reports   : "
        f"{len(unique_reports):,}"
    )

    print(
        f"Unique suspect drugs    : "
        f"{len(unique_drugs):,}"
    )

    print(
        f"Unique adverse events   : "
        f"{len(unique_events):,}"
    )

    print(
        f"Unique drug-event pairs : "
        f"{len(unique_pairs):,}"
    )


    # ========================================================
    # CRITICAL MISSINGNESS
    # ========================================================

    print_subheader(
        "CRITICAL FIELD MISSINGNESS"
    )


    for column in CRITICAL_COLUMNS:

        count = (
            missing_critical[
                column
            ]
        )

        percent = safe_percent(
            count,
            rows_processed
        )


        print(
            f"{column:<20}: "
            f"{count:>12,} "
            f"({percent:.4f}%)"
        )


    # ========================================================
    # DUPLICATES
    # ========================================================

    print_subheader(
        "DUPLICATE OBSERVATIONS"
    )


    print(
        "Duplicate key:"
    )

    print(
        "safetyreportid + "
        "suspect_drug + "
        "adverse_event"
    )


    print(
        f"\nDuplicate rows: "
        f"{duplicate_observations:,}"
    )


    if duplicate_observations == 0:

        print(
            "\n✅ No duplicate "
            "report-drug-event observations."
        )

    else:

        print(
            "\n⚠️ Duplicate observations "
            "exist and must be resolved "
            "before signal calculation."
        )


    # ========================================================
    # YEAR DISTRIBUTION
    # ========================================================

    print_subheader(
        "YEAR DISTRIBUTION"
    )


    for key in sorted(
        year_counter
    ):

        count = (
            year_counter[key]
        )

        percent = safe_percent(
            count,
            rows_processed
        )


        print(
            f"{key:<10} "
            f"{count:>12,} "
            f"{percent:>8.2f}%"
        )


    # ========================================================
    # QUARTER DISTRIBUTION
    # ========================================================

    print_subheader(
        "QUARTER DISTRIBUTION"
    )


    for key in sorted(
        quarter_counter
    ):

        count = (
            quarter_counter[key]
        )

        percent = safe_percent(
            count,
            rows_processed
        )


        print(
            f"{key:<10} "
            f"{count:>12,} "
            f"{percent:>8.2f}%"
        )


    # ========================================================
    # SERIOUS DISTRIBUTION
    # ========================================================

    print_subheader(
        "SERIOUS DISTRIBUTION"
    )


    for key, count in (
        serious_counter
        .most_common()
    ):

        percent = safe_percent(
            count,
            rows_processed
        )


        print(
            f"{key:<12} "
            f"{count:>12,} "
            f"{percent:>8.2f}%"
        )


    # ========================================================
    # SEX DISTRIBUTION
    # ========================================================

    print_subheader(
        "PATIENT SEX DISTRIBUTION"
    )


    for key, count in (
        sex_counter
        .most_common()
    ):

        percent = safe_percent(
            count,
            rows_processed
        )


        print(
            f"{key:<12} "
            f"{count:>12,} "
            f"{percent:>8.2f}%"
        )


    # ========================================================
    # TOP COUNTRIES
    # ========================================================

    print_subheader(
        "TOP 15 COUNTRIES"
    )


    for key, count in (
        country_counter
        .most_common(15)
    ):

        percent = safe_percent(
            count,
            rows_processed
        )


        print(
            f"{key:<15} "
            f"{count:>12,} "
            f"{percent:>8.2f}%"
        )


    # ========================================================
    # AGE
    # ========================================================

    print_subheader(
        "AGE VALIDATION"
    )


    print(
        f"Non-missing age : "
        f"{age_count:,}"
    )

    print(
        f"Missing age     : "
        f"{age_missing:,}"
    )


    if age_count > 0:

        print(
            f"Mean age        : "
            f"{age_sum / age_count:.2f}"
        )

        print(
            f"Minimum age     : "
            f"{age_min:.4f}"
        )

        print(
            f"Maximum age     : "
            f"{age_max:.4f}"
        )


    print(
        f"Invalid age "
        f"(<0 or >120)    : "
        f"{invalid_age_count:,}"
    )


    # ========================================================
    # COMPLEXITY FEATURES
    # ========================================================

    print_subheader(
        "REPORT COMPLEXITY VALIDATION"
    )


    print(
        f"Maximum num_drugs      : "
        f"{max_num_drugs:,}"
    )

    print(
        f"Maximum num_reactions  : "
        f"{max_num_reactions:,}"
    )

    print(
        f"Rows num_drugs <= 0    : "
        f"{zero_drug_count:,}"
    )

    print(
        f"Rows num_reactions <= 0: "
        f"{zero_reaction_count:,}"
    )

    print(
        f"Rows num_drugs > 100   : "
        f"{very_high_drug_count:,}"
    )

    print(
        f"Rows reactions > 50    : "
        f"{very_high_reaction_count:,}"
    )


    # ========================================================
    # CLEAN LARGE SETS
    # ========================================================

    del unique_reports
    del unique_drugs
    del unique_events
    del unique_pairs
    del seen_observations

    gc.collect()


    return results


# ============================================================
# 12. VALIDATION DECISION
# ============================================================

def print_validation_decision(
    results
):

    print_subheader(
        "VALIDATION DECISION"
    )


    problems = []


    if (
        results[
            "missing_safetyreportid"
        ] > 0
    ):

        problems.append(
            "Missing safetyreportid values"
        )


    if (
        results[
            "missing_suspect_drug"
        ] > 0
    ):

        problems.append(
            "Missing suspect drug values"
        )


    if (
        results[
            "missing_adverse_event"
        ] > 0
    ):

        problems.append(
            "Missing adverse event values"
        )


    if (
        results[
            "duplicate_report_drug_event_rows"
        ] > 0
    ):

        problems.append(
            "Duplicate report-drug-event rows"
        )


    if (
        results[
            "zero_num_drugs"
        ] > 0
    ):

        problems.append(
            "Rows with zero drugs"
        )


    if (
        results[
            "zero_num_reactions"
        ] > 0
    ):

        problems.append(
            "Rows with zero reactions"
        )


    if problems:

        print(
            "⚠️ REVIEW REQUIRED"
        )

        print(
            "\nThe following issues "
            "were detected:"
        )


        for problem in problems:

            print(
                f"  - {problem}"
            )


    else:

        print(
            "✅ PASS"
        )

        print(
            "\nCanonical drug-event "
            "structure is valid for the "
            "next signal-building stage."
        )


# ============================================================
# 13. MAIN VALIDATION
# ============================================================

def main():

    print_header(
        "PharmaAI - Canonical FAERS Dataset Validation"
    )


    print(
        f"\nProject:"
        f"\n{PROJECT_ROOT}"
    )


    print(
        f"\nBatch size: "
        f"{BATCH_SIZE:,}"
    )


    all_results = []


    # ========================================================
    # DATASET LOOP
    # ========================================================

    for dataset_name, file_path in (
        DATASETS.items()
    ):


        print_header(
            dataset_name
        )


        # ----------------------------------------------------
        # File
        # ----------------------------------------------------

        exists = validate_file_exists(
            dataset_name,
            file_path
        )


        if not exists:

            continue


        # ----------------------------------------------------
        # Open metadata
        # ----------------------------------------------------

        parquet_file = pq.ParquetFile(
            file_path
        )


        # ----------------------------------------------------
        # Schema
        # ----------------------------------------------------

        (
            missing_columns,
            unexpected_columns
        ) = validate_schema(
            parquet_file
        )


        if missing_columns:

            print(
                "\n❌ Validation stopped "
                "for this dataset because "
                "required schema fields "
                "are missing."
            )

            continue


        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        metadata_rows = (
            validate_metadata(
                dataset_name,
                parquet_file
            )
        )


        # ----------------------------------------------------
        # Streaming validation
        # ----------------------------------------------------

        results = stream_validate(

            dataset_name,

            file_path,

            parquet_file.schema.names
        )


        results[
            "metadata_rows"
        ] = metadata_rows


        results[
            "row_count_matches"
        ] = (

            metadata_rows
            ==
            EXPECTED_ROWS.get(
                dataset_name,
                metadata_rows
            )
        )


        # ----------------------------------------------------
        # Decision
        # ----------------------------------------------------

        print_validation_decision(
            results
        )


        all_results.append(
            results
        )


        del parquet_file

        gc.collect()


    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    if all_results:

        summary_df = pd.DataFrame(
            all_results
        )


        summary_path = (
            RESULTS_DIR
            / "canonical_validation_summary.csv"
        )


        summary_df.to_csv(
            summary_path,
            index=False
        )


        print_header(
            "FINAL VALIDATION SUMMARY"
        )


        display_columns = [

            "dataset",

            "rows",

            "unique_reports",

            "unique_suspect_drugs",

            "unique_adverse_events",

            "unique_drug_event_pairs",

            "duplicate_report_drug_event_rows",
        ]


        print(
            summary_df[
                display_columns
            ].to_string(
                index=False
            )
        )


        print(
            f"\nValidation summary saved:"
            f"\n{summary_path}"
        )


    print("\n" + "=" * 78)

    print(
        "PharmaAI validation finished."
    )

    print("=" * 78)


# ============================================================
# 14. RUN
# ============================================================

if __name__ == "__main__":

    try:

        main()


    except KeyboardInterrupt:

        print(
            "\n\nValidation interrupted "
            "by user."
        )

        sys.exit(1)


    except Exception as error:

        print(
            "\n❌ Unexpected validation "
            "error:"
        )

        print(
            error
        )

        raise