from pathlib import Path
import gc
import math
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


# ============================================================
# PharmaAI - Build Pharmacovigilance Signal Table
# ============================================================
#
# INPUT
# -----
# Canonical report × suspect-drug × adverse-event observations
#
# OUTPUT
# ------
# One row per:
#
#       suspect_drug × adverse_event
#
# containing:
#
#   pair_report_count
#   drug_report_count
#   event_report_count
#   total_reports_in_signal_universe
#
#   a
#   b
#   c
#   d
#
#   ROR
#   ROR 95% CI
#   PRR
#   IC
#
# IMPORTANT
# ---------
# Contingency-table counts are based on UNIQUE SAFETY REPORTS,
# not on the number of expanded drug-event rows.
#
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\mamid\OneDrive\Desktop\PharmaAI"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

SIGNAL_DIR = (
    PROCESSED_DIR
    / "signals"
)

SIGNAL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "signals"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. INPUT DATASETS
# ============================================================

DATASETS = {

    "2025": {
        "input":
            PROCESSED_DIR
            / "faers_2025_clean.parquet",

        "output":
            SIGNAL_DIR
            / "drug_event_signals_2025.parquet",
    },

    "2026_Q1": {
        "input":
            PROCESSED_DIR
            / "faers_2026_Q1_clean.parquet",

        "output":
            SIGNAL_DIR
            / "drug_event_signals_2026_Q1.parquet",
    },
}


# ============================================================
# 3. OPTIONS
# ============================================================

# Process both 2025 and 2026 Q1.
PROCESS_DATASETS = [
    "2025",
    "2026_Q1",
]


# Minimum pair count is NOT applied to the canonical output.
#
# We preserve all pairs and add a Boolean eligibility column.
MIN_PAIR_REPORTS = 5


# Number of rows shown in terminal.
TOP_N = 20


# ============================================================
# 4. HELPER
# ============================================================

def print_header(title):

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_subheader(title):

    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)


# ============================================================
# 5. READ CORE OBSERVATIONS
# ============================================================

def read_core_columns(file_path):

    print_subheader(
        "READING CANONICAL DATA"
    )

    print(
        f"Input:\n{file_path}"
    )

    parquet_file = pq.ParquetFile(
        file_path
    )

    required = [
        "safetyreportid",
        "suspect_drug",
        "adverse_event",
    ]

    available = set(
        parquet_file.schema.names
    )

    missing = [
        column
        for column in required
        if column not in available
    ]

    if missing:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )


    # Only read the three columns required for the
    # traditional signal calculations.
    table = pq.read_table(
        file_path,
        columns=required
    )


    print(
        f"Expanded observations : "
        f"{table.num_rows:,}"
    )

    return table


# ============================================================
# 6. GLOBAL DEDUPLICATION
# ============================================================

def remove_duplicate_observations(table):

    print_subheader(
        "REPORT-DRUG-EVENT DEDUPLICATION"
    )

    before = table.num_rows


    # Convert only the three-column table to pandas.
    #
    # 2025 has ~16M rows, but these are only the key columns.
    # This is substantially smaller than loading the full
    # canonical dataset with all metadata.
    df = table.to_pandas()


    del table
    gc.collect()


    df = df.drop_duplicates(
        subset=[
            "safetyreportid",
            "suspect_drug",
            "adverse_event",
        ]
    )


    after = len(df)

    removed = before - after


    print(
        f"Before : {before:,}"
    )

    print(
        f"After  : {after:,}"
    )

    print(
        f"Removed: {removed:,}"
    )


    return df


# ============================================================
# 7. BUILD REPORT COUNTS
# ============================================================

def build_counts(df):

    print_subheader(
        "BUILDING REPORT-LEVEL COUNTS"
    )


    # ========================================================
    # TOTAL UNIQUE REPORTS
    # ========================================================

    total_reports = (
        df["safetyreportid"]
        .nunique()
    )


    print(
        f"Total unique safety reports: "
        f"{total_reports:,}"
    )


    # ========================================================
    # PAIR REPORT COUNT
    # ========================================================
    #
    # a = reports containing both drug D and event E
    #
    # Because report-drug-event rows are already unique,
    # group size is the report count.
    # ========================================================

    print(
        "Calculating pair report counts..."
    )


    pair_counts = (

        df.groupby(
            [
                "suspect_drug",
                "adverse_event"
            ],
            observed=True,
            sort=False
        )
        .size()
        .rename(
            "pair_report_count"
        )
        .reset_index()
    )


    print(
        f"Unique drug-event pairs: "
        f"{len(pair_counts):,}"
    )


    # ========================================================
    # DRUG REPORT COUNT
    # ========================================================
    #
    # N(D) = unique reports containing suspect drug D
    # ========================================================

    print(
        "Calculating drug report counts..."
    )


    drug_reports = (

        df[
            [
                "safetyreportid",
                "suspect_drug"
            ]
        ]
        .drop_duplicates()
        .groupby(
            "suspect_drug",
            observed=True,
            sort=False
        )
        .size()
        .rename(
            "drug_report_count"
        )
        .reset_index()
    )


    # ========================================================
    # EVENT REPORT COUNT
    # ========================================================
    #
    # N(E) = unique reports containing event E
    # ========================================================

    print(
        "Calculating event report counts..."
    )


    event_reports = (

        df[
            [
                "safetyreportid",
                "adverse_event"
            ]
        ]
        .drop_duplicates()
        .groupby(
            "adverse_event",
            observed=True,
            sort=False
        )
        .size()
        .rename(
            "event_report_count"
        )
        .reset_index()
    )


    # ========================================================
    # MERGE COUNTS
    # ========================================================

    print(
        "Merging counts..."
    )


    signal_df = pair_counts.merge(

        drug_reports,

        on="suspect_drug",

        how="left",

        validate="many_to_one"
    )


    signal_df = signal_df.merge(

        event_reports,

        on="adverse_event",

        how="left",

        validate="many_to_one"
    )


    signal_df[
        "total_reports_in_signal_universe"
    ] = total_reports


    del pair_counts
    del drug_reports
    del event_reports

    gc.collect()


    return (
        signal_df,
        total_reports
    )


# ============================================================
# 8. BUILD 2 × 2 CONTINGENCY TABLE
# ============================================================

def build_contingency_table(signal_df):

    print_subheader(
        "BUILDING 2 × 2 CONTINGENCY TABLE"
    )


    # --------------------------------------------------------
    # a = Drug AND Event
    # --------------------------------------------------------

    signal_df[
        "a_drug_and_event"
    ] = (
        signal_df[
            "pair_report_count"
        ]
        .astype("int64")
    )


    # --------------------------------------------------------
    # b = Drug AND NOT Event
    #
    # N(D) - a
    # --------------------------------------------------------

    signal_df[
        "b_drug_other_event"
    ] = (

        signal_df[
            "drug_report_count"
        ]

        -

        signal_df[
            "a_drug_and_event"
        ]
    )


    # --------------------------------------------------------
    # c = NOT Drug AND Event
    #
    # N(E) - a
    # --------------------------------------------------------

    signal_df[
        "c_other_drug_event"
    ] = (

        signal_df[
            "event_report_count"
        ]

        -

        signal_df[
            "a_drug_and_event"
        ]
    )


    # --------------------------------------------------------
    # d = NOT Drug AND NOT Event
    #
    # N - a - b - c
    #
    # Equivalent:
    #
    # N - N(D) - N(E) + a
    # --------------------------------------------------------

    signal_df[
        "d_other_drug_other_event"
    ] = (

        signal_df[
            "total_reports_in_signal_universe"
        ]

        -

        signal_df[
            "drug_report_count"
        ]

        -

        signal_df[
            "event_report_count"
        ]

        +

        signal_df[
            "a_drug_and_event"
        ]
    )


    # ========================================================
    # VALIDATE COUNTS
    # ========================================================

    count_columns = [

        "a_drug_and_event",
        "b_drug_other_event",
        "c_other_drug_event",
        "d_other_drug_other_event",
    ]


    negative_counts = (

        signal_df[
            count_columns
        ]

        < 0

    ).any(
        axis=1
    ).sum()


    print(
        f"Pairs with negative "
        f"contingency counts: "
        f"{negative_counts:,}"
    )


    # Check:
    #
    # a + b + c + d = N
    # --------------------------------------------------------

    totals = (

        signal_df[
            "a_drug_and_event"
        ]

        +

        signal_df[
            "b_drug_other_event"
        ]

        +

        signal_df[
            "c_other_drug_event"
        ]

        +

        signal_df[
            "d_other_drug_other_event"
        ]
    )


    invalid_totals = (

        totals

        !=

        signal_df[
            "total_reports_in_signal_universe"
        ]

    ).sum()


    print(
        f"Pairs failing "
        f"a+b+c+d=N: "
        f"{invalid_totals:,}"
    )


    if (
        negative_counts > 0
        or
        invalid_totals > 0
    ):

        raise ValueError(
            "Invalid contingency table "
            "counts detected."
        )


    print(
        "✅ Contingency-table validation passed."
    )


    return signal_df


# ============================================================
# 9. CONTINUITY CORRECTION
# ============================================================

def get_corrected_counts(signal_df):

    """
    Apply Haldane-Anscombe correction only when one or more
    contingency cells are zero.

    This prevents infinite/undefined ROR estimates.

    Original a,b,c,d remain unchanged in the output.
    """

    a = (
        signal_df[
            "a_drug_and_event"
        ]
        .astype("float64")
        .to_numpy()
    )

    b = (
        signal_df[
            "b_drug_other_event"
        ]
        .astype("float64")
        .to_numpy()
    )

    c = (
        signal_df[
            "c_other_drug_event"
        ]
        .astype("float64")
        .to_numpy()
    )

    d = (
        signal_df[
            "d_other_drug_other_event"
        ]
        .astype("float64")
        .to_numpy()
    )


    zero_cell = (
        (a == 0)
        |
        (b == 0)
        |
        (c == 0)
        |
        (d == 0)
    )


    a_corr = a.copy()
    b_corr = b.copy()
    c_corr = c.copy()
    d_corr = d.copy()


    a_corr[
        zero_cell
    ] += 0.5

    b_corr[
        zero_cell
    ] += 0.5

    c_corr[
        zero_cell
    ] += 0.5

    d_corr[
        zero_cell
    ] += 0.5


    return (
        a,
        b,
        c,
        d,
        a_corr,
        b_corr,
        c_corr,
        d_corr,
        zero_cell
    )


# ============================================================
# 10. CALCULATE ROR
# ============================================================

def calculate_ror(signal_df):

    print_subheader(
        "CALCULATING ROR"
    )


    (
        a,
        b,
        c,
        d,
        a_corr,
        b_corr,
        c_corr,
        d_corr,
        zero_cell
    ) = get_corrected_counts(
        signal_df
    )


    # ========================================================
    # ROR
    #
    #       a / b
    # ROR = -----
    #       c / d
    #
    #     = ad / bc
    # ========================================================

    ror = (

        a_corr
        *
        d_corr

    ) / (

        b_corr
        *
        c_corr
    )


    # ========================================================
    # Standard error of log(ROR)
    # ========================================================

    se_log_ror = np.sqrt(

        (1 / a_corr)

        +

        (1 / b_corr)

        +

        (1 / c_corr)

        +

        (1 / d_corr)
    )


    log_ror = np.log(
        ror
    )


    # ========================================================
    # 95% CI
    # ========================================================

    lower = np.exp(

        log_ror

        -

        1.96
        *
        se_log_ror
    )


    upper = np.exp(

        log_ror

        +

        1.96
        *
        se_log_ror
    )


    signal_df[
        "ror"
    ] = ror


    signal_df[
        "ror_ci_lower"
    ] = lower


    signal_df[
        "ror_ci_upper"
    ] = upper


    signal_df[
        "continuity_correction_applied"
    ] = zero_cell


    print(
        f"Pairs requiring 0.5 "
        f"continuity correction: "
        f"{int(zero_cell.sum()):,}"
    )


    return signal_df


# ============================================================
# 11. CALCULATE PRR
# ============================================================

def calculate_prr(signal_df):

    print_subheader(
        "CALCULATING PRR"
    )


    a = (
        signal_df[
            "a_drug_and_event"
        ]
        .astype("float64")
        .to_numpy()
    )

    b = (
        signal_df[
            "b_drug_other_event"
        ]
        .astype("float64")
        .to_numpy()
    )

    c = (
        signal_df[
            "c_other_drug_event"
        ]
        .astype("float64")
        .to_numpy()
    )

    d = (
        signal_df[
            "d_other_drug_other_event"
        ]
        .astype("float64")
        .to_numpy()
    )


    # ========================================================
    # PRR
    #
    #        a/(a+b)
    # PRR = -----------
    #        c/(c+d)
    # ========================================================

    drug_event_rate = np.divide(

        a,

        a + b,

        out=np.full_like(
            a,
            np.nan
        ),

        where=(
            (a + b) != 0
        )
    )


    background_event_rate = np.divide(

        c,

        c + d,

        out=np.full_like(
            c,
            np.nan
        ),

        where=(
            (c + d) != 0
        )
    )


    prr = np.divide(

        drug_event_rate,

        background_event_rate,

        out=np.full_like(
            drug_event_rate,
            np.nan
        ),

        where=(
            background_event_rate
            != 0
        )
    )


    signal_df[
        "prr"
    ] = prr


    return signal_df


# ============================================================
# 12. CALCULATE INFORMATION COMPONENT
# ============================================================

def calculate_ic(signal_df):

    print_subheader(
        "CALCULATING IC"
    )


    # ========================================================
    # Observed
    # ========================================================

    observed = (
        signal_df[
            "a_drug_and_event"
        ]
        .astype("float64")
        .to_numpy()
    )


    drug_reports = (
        signal_df[
            "drug_report_count"
        ]
        .astype("float64")
        .to_numpy()
    )


    event_reports = (
        signal_df[
            "event_report_count"
        ]
        .astype("float64")
        .to_numpy()
    )


    total_reports = (
        signal_df[
            "total_reports_in_signal_universe"
        ]
        .astype("float64")
        .to_numpy()
    )


    # ========================================================
    # Expected count under independence
    #
    # E = N(D) × N(E) / N
    # ========================================================

    expected = (

        drug_reports
        *
        event_reports

    ) / total_reports


    # ========================================================
    # Information Component
    #
    # IC = log2(observed / expected)
    #
    # Since all canonical pairs have a >= 1, observed is
    # positive.
    # ========================================================

    ic = np.log2(

        observed
        /
        expected
    )


    signal_df[
        "expected_pair_count"
    ] = expected


    signal_df[
        "ic"
    ] = ic


    return signal_df


# ============================================================
# 13. SUPPORT / SIGNAL FLAGS
# ============================================================

def add_flags(signal_df):

    print_subheader(
        "ADDING ANALYSIS FLAGS"
    )


    # ========================================================
    # Minimum support
    # ========================================================

    signal_df[
        "meets_minimum_support"
    ] = (

        signal_df[
            "pair_report_count"
        ]

        >=

        MIN_PAIR_REPORTS
    )


    # ========================================================
    # Simple ROR screening flag
    #
    # We keep this separate from the ML ground truth.
    #
    # This is NOT the ML label.
    # ========================================================

    signal_df[
        "ror_lower_ci_gt_1"
    ] = (

        signal_df[
            "ror_ci_lower"
        ]

        > 1
    )


    # ========================================================
    # PRR descriptive screening flag
    #
    # Again: NOT the ML ground truth.
    # ========================================================

    signal_df[
        "prr_gt_2"
    ] = (

        signal_df[
            "prr"
        ]

        > 2
    )


    print(
        f"Minimum support threshold: "
        f"{MIN_PAIR_REPORTS}"
    )


    supported = int(
        signal_df[
            "meets_minimum_support"
        ].sum()
    )


    print(
        f"Pairs meeting support: "
        f"{supported:,}"
    )


    print(
        f"Pairs below support  : "
        f"{len(signal_df) - supported:,}"
    )


    return signal_df


# ============================================================
# 14. FINAL QUALITY CHECK
# ============================================================

def final_quality_check(signal_df):

    print_subheader(
        "FINAL SIGNAL-TABLE VALIDATION"
    )


    print(
        f"Rows / unique pairs: "
        f"{len(signal_df):,}"
    )


    duplicate_pairs = (

        signal_df.duplicated(
            subset=[
                "suspect_drug",
                "adverse_event"
            ]
        )
        .sum()
    )


    print(
        f"Duplicate drug-event pairs: "
        f"{duplicate_pairs:,}"
    )


    missing_counts = (

        signal_df[
            [
                "pair_report_count",
                "drug_report_count",
                "event_report_count",
                "total_reports_in_signal_universe",
                "ror",
                "ror_ci_lower",
                "ror_ci_upper",
                "prr",
                "ic",
            ]
        ]
        .isna()
        .sum()
    )


    print(
        "\nMissing values:"
    )

    print(
        missing_counts.to_string()
    )


    # --------------------------------------------------------
    # Mathematical checks
    # --------------------------------------------------------

    invalid_a = (

        signal_df[
            "a_drug_and_event"
        ]

        !=

        signal_df[
            "pair_report_count"
        ]

    ).sum()


    invalid_drug = (

        (
            signal_df[
                "a_drug_and_event"
            ]

            +

            signal_df[
                "b_drug_other_event"
            ]
        )

        !=

        signal_df[
            "drug_report_count"
        ]

    ).sum()


    invalid_event = (

        (
            signal_df[
                "a_drug_and_event"
            ]

            +

            signal_df[
                "c_other_drug_event"
            ]
        )

        !=

        signal_df[
            "event_report_count"
        ]

    ).sum()


    print(
        f"\na != pair_report_count : "
        f"{invalid_a:,}"
    )

    print(
        f"a+b != drug reports     : "
        f"{invalid_drug:,}"
    )

    print(
        f"a+c != event reports    : "
        f"{invalid_event:,}"
    )


    if (
        duplicate_pairs == 0
        and
        invalid_a == 0
        and
        invalid_drug == 0
        and
        invalid_event == 0
    ):

        print(
            "\n✅ Signal table structural "
            "validation passed."
        )

    else:

        raise ValueError(
            "Signal-table validation failed."
        )


# ============================================================
# 15. DISPLAY TOP SIGNALS
# ============================================================

def show_top_signals(signal_df):

    print_subheader(
        f"TOP {TOP_N} SUPPORTED SIGNALS BY ROR"
    )


    supported = signal_df[

        signal_df[
            "meets_minimum_support"
        ]

    ].copy()


    top = (

        supported
        .sort_values(
            [
                "ror_ci_lower",
                "pair_report_count"
            ],
            ascending=[
                False,
                False
            ]
        )
        .head(
            TOP_N
        )
    )


    columns = [

        "suspect_drug",
        "adverse_event",
        "pair_report_count",
        "ror",
        "ror_ci_lower",
        "ror_ci_upper",
        "prr",
        "ic",
    ]


    with pd.option_context(
        "display.max_columns",
        None,

        "display.width",
        200,

        "display.float_format",
        "{:.4f}".format
    ):

        print(
            top[
                columns
            ].to_string(
                index=False
            )
        )


# ============================================================
# 16. SAVE OUTPUT
# ============================================================

def save_signal_table(
    signal_df,
    output_file,
    dataset_name
):

    print_subheader(
        "SAVING SIGNAL TABLE"
    )


    signal_df = signal_df.sort_values(

        by=[
            "pair_report_count",
            "ror_ci_lower"
        ],

        ascending=[
            False,
            False
        ]
    ).reset_index(
        drop=True
    )


    signal_df.to_parquet(

        output_file,

        index=False,

        engine="pyarrow",

        compression="snappy"
    )


    size_mb = (

        output_file
        .stat()
        .st_size

        /

        (1024 ** 2)
    )


    print(
        f"Saved:\n{output_file}"
    )

    print(
        f"Rows     : "
        f"{len(signal_df):,}"
    )

    print(
        f"File size: "
        f"{size_mb:,.2f} MB"
    )


    # ========================================================
    # SUMMARY CSV
    # ========================================================

    summary = pd.DataFrame(
        [
            {
                "dataset":
                    dataset_name,

                "total_drug_event_pairs":
                    len(signal_df),

                "pairs_meeting_min_support":
                    int(
                        signal_df[
                            "meets_minimum_support"
                        ].sum()
                    ),

                "pairs_ror_lower_ci_gt_1":
                    int(
                        signal_df[
                            "ror_lower_ci_gt_1"
                        ].sum()
                    ),

                "pairs_prr_gt_2":
                    int(
                        signal_df[
                            "prr_gt_2"
                        ].sum()
                    ),

                "median_pair_report_count":
                    float(
                        signal_df[
                            "pair_report_count"
                        ].median()
                    ),

                "median_ror":
                    float(
                        signal_df[
                            "ror"
                        ].median()
                    ),

                "median_prr":
                    float(
                        signal_df[
                            "prr"
                        ].median()
                    ),

                "median_ic":
                    float(
                        signal_df[
                            "ic"
                        ].median()
                    ),
            }
        ]
    )


    summary_file = (

        RESULTS_DIR

        /

        f"signal_summary_{dataset_name}.csv"
    )


    summary.to_csv(
        summary_file,
        index=False
    )


    print(
        f"\nSummary:\n{summary_file}"
    )


# ============================================================
# 17. PROCESS ONE DATASET
# ============================================================

def process_dataset(
    dataset_name,
    config
):

    print_header(
        f"PHARMAAI SIGNAL TABLE - "
        f"{dataset_name}"
    )


    input_file = (
        config[
            "input"
        ]
    )

    output_file = (
        config[
            "output"
        ]
    )


    if not input_file.exists():

        raise FileNotFoundError(
            f"Input file not found:\n"
            f"{input_file}"
        )


    # ========================================================
    # READ
    # ========================================================

    table = read_core_columns(
        input_file
    )


    # ========================================================
    # GLOBAL DEDUPLICATION
    # ========================================================

    df = remove_duplicate_observations(
        table
    )


    # ========================================================
    # COUNTS
    # ========================================================

    signal_df, total_reports = (
        build_counts(
            df
        )
    )


    # We no longer need the 16M-row observation dataframe.
    del df

    gc.collect()


    # ========================================================
    # CONTINGENCY TABLE
    # ========================================================

    signal_df = (
        build_contingency_table(
            signal_df
        )
    )


    # ========================================================
    # ROR
    # ========================================================

    signal_df = (
        calculate_ror(
            signal_df
        )
    )


    # ========================================================
    # PRR
    # ========================================================

    signal_df = (
        calculate_prr(
            signal_df
        )
    )


    # ========================================================
    # IC
    # ========================================================

    signal_df = (
        calculate_ic(
            signal_df
        )
    )


    # ========================================================
    # FLAGS
    # ========================================================

    signal_df = add_flags(
        signal_df
    )


    # ========================================================
    # QUALITY CHECK
    # ========================================================

    final_quality_check(
        signal_df
    )


    # ========================================================
    # TOP SIGNALS
    # ========================================================

    show_top_signals(
        signal_df
    )


    # ========================================================
    # SAVE
    # ========================================================

    save_signal_table(

        signal_df,

        output_file,

        dataset_name
    )


    print_subheader(
        "DATASET COMPLETE"
    )


    print(
        f"Dataset              : "
        f"{dataset_name}"
    )

    print(
        f"Signal universe      : "
        f"{total_reports:,} reports"
    )

    print(
        f"Drug-event pairs     : "
        f"{len(signal_df):,}"
    )

    print(
        f"Pairs with >= "
        f"{MIN_PAIR_REPORTS} reports: "
        f"{int(signal_df['meets_minimum_support'].sum()):,}"
    )


    del signal_df

    gc.collect()


# ============================================================
# 18. MAIN
# ============================================================

def main():

    print_header(
        "PharmaAI - Traditional "
        "Pharmacovigilance Signal Construction"
    )


    print(
        "\nThis script calculates:"
    )

    print(
        "  • report-level pair counts"
    )

    print(
        "  • a, b, c, d"
    )

    print(
        "  • ROR"
    )

    print(
        "  • ROR 95% CI"
    )

    print(
        "  • PRR"
    )

    print(
        "  • IC"
    )


    print(
        "\nDatasets:"
    )


    for dataset in PROCESS_DATASETS:

        print(
            f"  • {dataset}"
        )


    # ========================================================
    # DATASET LOOP
    # ========================================================

    for dataset_name in PROCESS_DATASETS:


        if dataset_name not in DATASETS:

            print(
                f"\n⚠️ Unknown dataset: "
                f"{dataset_name}"
            )

            continue


        process_dataset(

            dataset_name,

            DATASETS[
                dataset_name
            ]
        )


    print_header(
        "PHARMAAI SIGNAL CONSTRUCTION COMPLETE"
    )


    print(
        "\nOutputs:"
    )


    for dataset_name in PROCESS_DATASETS:

        if dataset_name in DATASETS:

            print(
                f"\n{dataset_name}:"
            )

            print(
                DATASETS[
                    dataset_name
                ][
                    "output"
                ]
            )


# ============================================================
# 19. RUN
# ============================================================

if __name__ == "__main__":

    try:

        main()


    except KeyboardInterrupt:

        print(
            "\n\nSignal construction "
            "interrupted by user."
        )

        sys.exit(1)


    except Exception as error:

        print(
            "\n❌ Signal construction failed:"
        )

        print(
            error
        )

        raise