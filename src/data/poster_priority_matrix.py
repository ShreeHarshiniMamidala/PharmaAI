from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PHARMAAI — PRELIMINARY SIGNAL PRIORITY MATRIX
# ============================================================
#
# PURPOSE
# -------
# Create a poster-quality prioritization visualization for the
# first 50 supported drug-event associations produced by the
# preliminary signal-detection pipeline.
#
# Priority here means:
#
#     PRIORITY FOR FURTHER PHARMACOVIGILANCE REVIEW
#
# It DOES NOT mean:
#
#     - clinical risk
#     - confirmed adverse drug reaction
#     - causality
#     - regulatory safety signal
#
#
# INPUT
# -----
# results/preliminary/tables/
# └── preliminary_candidate_signals.csv
#
#
# FILES CREATED
# -------------
# results/preliminary/poster_graphs/
#
# ├── 06_priority_matrix_top50.png
# └── 06_priority_matrix_top50.pdf
#
#
# results/preliminary/tables/
#
# ├── priority_matrix_top50.csv
# └── priority_matrix_top10_poster.csv
#
#
# VISUAL ENCODING
# ---------------
#
# X-axis:
#     Number of reports (log scale)
#
# Y-axis:
#     IC
#
# Bubble size:
#     Conservative ROR evidence based on ROR lower 95% CI
#
# Priority quadrants:
#
#     Upper-right:
#         PRIORITY REVIEW
#
#     Upper-left:
#         STRONG SIGNAL / LOWER SUPPORT
#
#     Lower-right:
#         HIGH SUPPORT / MODERATE SIGNAL
#
#     Lower-left:
#         MONITOR
#
#
# QUADRANT BOUNDARIES
# -------------------
#
# Boundaries are calculated from the median report count and
# median IC among the 50 displayed associations.
#
# They are descriptive prioritization boundaries, NOT clinical
# or regulatory thresholds.
#
# ============================================================


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\mamid\OneDrive\Desktop\PharmaAI"
)

TABLE_DIR = (
    PROJECT_ROOT
    / "results"
    / "preliminary"
    / "tables"
)

GRAPH_DIR = (
    PROJECT_ROOT
    / "results"
    / "preliminary"
    / "poster_graphs"
)

GRAPH_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# INPUT
# ============================================================

INPUT_FILE = (
    TABLE_DIR
    / "preliminary_candidate_signals.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

OUTPUT_PNG = (
    GRAPH_DIR
    / "06_priority_matrix_top50.png"
)

OUTPUT_PDF = (
    GRAPH_DIR
    / "06_priority_matrix_top50.pdf"
)

OUTPUT_TOP50 = (
    TABLE_DIR
    / "priority_matrix_top50.csv"
)

OUTPUT_TOP10 = (
    TABLE_DIR
    / "priority_matrix_top10_poster.csv"
)


# ============================================================
# SETTINGS
# ============================================================

NUMBER_OF_ASSOCIATIONS = 50

NUMBER_OF_LABELS = 10

PNG_DPI = 600


# ============================================================
# POSTER FONT SETTINGS
# ============================================================

plt.rcParams.update({

    "figure.dpi": 150,

    "savefig.dpi": PNG_DPI,

    "font.size": 14,

    "axes.titlesize": 22,

    "axes.labelsize": 17,

    "xtick.labelsize": 13,

    "ytick.labelsize": 13,

    "legend.fontsize": 12,
})


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\n" + "=" * 78)
    print("PHARMAAI — PRELIMINARY SIGNAL PRIORITY MATRIX")
    print("=" * 78)

    print(
        f"\nReading:\n{INPUT_FILE}"
    )

    if not INPUT_FILE.exists():

        raise FileNotFoundError(

            f"\nCould not find:\n"
            f"{INPUT_FILE}\n\n"
            f"Run preliminary_signal_analysis_revised.py first."
        )


    required_columns = [

        "suspect_drug",

        "adverse_event",

        "pair_report_count",

        "ror",

        "ror_ci_lower",

        "ror_ci_upper",

        "prr",

        "ic",
    ]


    df = pd.read_csv(
        INPUT_FILE,
        usecols=required_columns
    )


    print(
        f"\nCandidate associations available: "
        f"{len(df):,}"
    )


    # --------------------------------------------------------
    # Convert numerical columns
    # --------------------------------------------------------

    numeric_columns = [

        "pair_report_count",

        "ror",

        "ror_ci_lower",

        "ror_ci_upper",

        "prr",

        "ic",
    ]


    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


    df = df.dropna(
        subset=numeric_columns
    ).copy()


    # --------------------------------------------------------
    # Take FIRST 50 from existing supported ranking
    # --------------------------------------------------------

    top50 = (
        df
        .head(NUMBER_OF_ASSOCIATIONS)
        .copy()
        .reset_index(drop=True)
    )


    if len(top50) < NUMBER_OF_ASSOCIATIONS:

        print(
            f"\nWARNING: Only {len(top50)} valid associations "
            f"were available."
        )


    top50["original_rank"] = (
        np.arange(1, len(top50) + 1)
    )


    print(
        f"\nAssociations selected for matrix: "
        f"{len(top50):,}"
    )


    return top50


# ============================================================
# CREATE PRIORITY VARIABLES
# ============================================================

def build_priority_variables(df):

    print("\n" + "=" * 78)
    print("BUILDING PRIORITY VARIABLES")
    print("=" * 78)


    # ========================================================
    # X AXIS
    # ========================================================
    #
    # Number of reports spans a wide range, so use log10.
    # ========================================================

    df["log10_report_count"] = np.log10(
        df["pair_report_count"]
    )


    # ========================================================
    # QUADRANT BOUNDARIES
    # ========================================================

    report_boundary = df[
        "pair_report_count"
    ].median()


    ic_boundary = df[
        "ic"
    ].median()


    log_report_boundary = np.log10(
        report_boundary
    )


    print(
        f"\nMedian report-count boundary: "
        f"{report_boundary:,.1f}"
    )


    print(
        f"Median IC boundary: "
        f"{ic_boundary:.3f}"
    )


    # ========================================================
    # PRIORITY REGION
    # ========================================================

    conditions = [

        (
            (df["pair_report_count"] >= report_boundary)
            &
            (df["ic"] >= ic_boundary)
        ),

        (
            (df["pair_report_count"] < report_boundary)
            &
            (df["ic"] >= ic_boundary)
        ),

        (
            (df["pair_report_count"] >= report_boundary)
            &
            (df["ic"] < ic_boundary)
        ),
    ]


    categories = [

        "Priority Review",

        "Strong Signal / Lower Support",

        "High Support / Moderate Signal",
    ]


    df["priority_region"] = np.select(

        conditions,

        categories,

        default="Monitor"
    )


    # ========================================================
    # BUBBLE SIZE
    # ========================================================
    #
    # ROR lower CI is used instead of raw ROR because it is
    # more conservative.
    #
    # log1p prevents very large ROR values from producing
    # enormous bubbles.
    # ========================================================

    conservative_strength = np.log1p(
        df["ror_ci_lower"]
    )


    min_strength = conservative_strength.min()

    max_strength = conservative_strength.max()


    if max_strength > min_strength:

        normalized_strength = (

            (
                conservative_strength
                -
                min_strength
            )

            /

            (
                max_strength
                -
                min_strength
            )
        )

    else:

        normalized_strength = pd.Series(
            np.ones(len(df)),
            index=df.index
        )


    df["bubble_size"] = (

        180

        +

        normalized_strength * 900
    )


    # ========================================================
    # PRIORITY ORDER
    # ========================================================
    #
    # This is a transparent descriptive ranking.
    #
    # Priority Review first, then:
    #
    #     IC descending
    #     report support descending
    #     conservative ROR descending
    #
    # ========================================================

    region_order = {

        "Priority Review": 1,

        "Strong Signal / Lower Support": 2,

        "High Support / Moderate Signal": 3,

        "Monitor": 4,
    }


    df["priority_region_order"] = (
        df["priority_region"]
        .map(region_order)
    )


    df = (
        df
        .sort_values(

            [

                "priority_region_order",

                "ic",

                "pair_report_count",

                "ror_ci_lower",
            ],

            ascending=[

                True,

                False,

                False,

                False,
            ]
        )
        .reset_index(drop=True)
    )


    df["review_priority_rank"] = (
        np.arange(1, len(df) + 1)
    )


    return (
        df,
        report_boundary,
        ic_boundary,
        log_report_boundary
    )


# ============================================================
# CREATE SHORT LABEL
# ============================================================

def make_label(drug, event):

    drug = str(drug).strip()

    event = str(event).strip()


    # Keep drug names reasonably short
    if len(drug) > 20:

        drug = drug[:18] + "…"


    # Wrap event text
    event = textwrap.fill(
        event.title(),
        width=24
    )


    return (
        f"{drug}\n{event}"
    )


# ============================================================
# CREATE PRIORITY MATRIX
# ============================================================

def create_priority_matrix(

    df,

    report_boundary,

    ic_boundary,

    log_report_boundary
):

    print("\n" + "=" * 78)
    print("CREATING PRIORITY MATRIX")
    print("=" * 78)


    # ========================================================
    # FIGURE
    # ========================================================

    fig, ax = plt.subplots(

        figsize=(16, 11)
    )


    # ========================================================
    # AXIS LIMITS
    # ========================================================

    x_min = (
        df["log10_report_count"].min()
        - 0.08
    )

    x_max = (
        df["log10_report_count"].max()
        + 0.08
    )


    y_range = (
        df["ic"].max()
        -
        df["ic"].min()
    )


    y_padding = max(
        y_range * 0.12,
        0.25
    )


    y_min = (
        df["ic"].min()
        -
        y_padding
    )

    y_max = (
        df["ic"].max()
        +
        y_padding
    )


    ax.set_xlim(
        x_min,
        x_max
    )


    ax.set_ylim(
        y_min,
        y_max
    )


    # ========================================================
    # QUADRANT BACKGROUNDS
    # ========================================================

    ax.axvspan(

        x_min,

        log_report_boundary,

        ymin=0,

        ymax=(
            (ic_boundary - y_min)
            /
            (y_max - y_min)
        ),

        alpha=0.05
    )


    ax.axvspan(

        log_report_boundary,

        x_max,

        ymin=0,

        ymax=(
            (ic_boundary - y_min)
            /
            (y_max - y_min)
        ),

        alpha=0.08
    )


    ax.axvspan(

        x_min,

        log_report_boundary,

        ymin=(
            (ic_boundary - y_min)
            /
            (y_max - y_min)
        ),

        ymax=1,

        alpha=0.08
    )


    ax.axvspan(

        log_report_boundary,

        x_max,

        ymin=(
            (ic_boundary - y_min)
            /
            (y_max - y_min)
        ),

        ymax=1,

        alpha=0.12
    )


    # ========================================================
    # QUADRANT BOUNDARY LINES
    # ========================================================

    ax.axvline(

        log_report_boundary,

        linestyle="--",

        linewidth=1.5
    )


    ax.axhline(

        ic_boundary,

        linestyle="--",

        linewidth=1.5
    )


    # ========================================================
    # PLOT ASSOCIATIONS BY REGION
    # ========================================================

    region_names = [

        "Monitor",

        "High Support / Moderate Signal",

        "Strong Signal / Lower Support",

        "Priority Review",
    ]


    markers = {

        "Monitor": "o",

        "High Support / Moderate Signal": "s",

        "Strong Signal / Lower Support": "^",

        "Priority Review": "D",
    }


    for region in region_names:

        subset = df[
            df["priority_region"] == region
        ]


        if subset.empty:

            continue


        ax.scatter(

            subset["log10_report_count"],

            subset["ic"],

            s=subset["bubble_size"],

            alpha=0.68,

            marker=markers[region],

            edgecolors="white",

            linewidths=1.0,

            label=region
        )


    # ========================================================
    # LABEL TOP 10
    # ========================================================

    top_labels = (
        df
        .head(NUMBER_OF_LABELS)
        .copy()
    )


    # Different offsets reduce label overlap.
    offsets = [

        (14, 16),

        (14, -34),

        (-145, 18),

        (-145, -38),

        (18, 28),

        (18, -46),

        (-150, 32),

        (-150, -50),

        (20, 40),

        (-150, 44),
    ]


    for position, (_, row) in enumerate(
        top_labels.iterrows()
    ):

        offset = offsets[
            position % len(offsets)
        ]


        label = make_label(

            row["suspect_drug"],

            row["adverse_event"]
        )


        ax.annotate(

            label,

            (
                row["log10_report_count"],
                row["ic"]
            ),

            xytext=offset,

            textcoords="offset points",

            fontsize=10,

            fontweight="bold",

            bbox=dict(

                boxstyle="round,pad=0.35",

                facecolor="white",

                alpha=0.88,

                edgecolor="0.75"
            ),

            arrowprops=dict(

                arrowstyle="-",

                linewidth=0.8,

                alpha=0.7
            )
        )


    # ========================================================
    # QUADRANT TITLES
    # ========================================================

    left_x = (
        x_min
        +
        0.03 * (x_max - x_min)
    )


    right_x = (
        log_report_boundary
        +
        0.03 * (x_max - x_min)
    )


    upper_y = (
        y_max
        -
        0.06 * (y_max - y_min)
    )


    lower_y = (
        ic_boundary
        -
        0.08 * (y_max - y_min)
    )


    ax.text(

        left_x,

        upper_y,

        "STRONG SIGNAL\nLOWER SUPPORT",

        fontsize=13,

        fontweight="bold",

        va="top"
    )


    ax.text(

        right_x,

        upper_y,

        "PRIORITY REVIEW",

        fontsize=15,

        fontweight="bold",

        va="top"
    )


    ax.text(

        left_x,

        lower_y,

        "MONITOR",

        fontsize=13,

        fontweight="bold",

        va="top"
    )


    ax.text(

        right_x,

        lower_y,

        "HIGH SUPPORT\nMODERATE SIGNAL",

        fontsize=13,

        fontweight="bold",

        va="top"
    )


    # ========================================================
    # X AXIS TICKS
    # ========================================================
    #
    # Coordinates are log10(report count), but labels show
    # original report counts.
    # ========================================================

    possible_report_ticks = [

        10,

        20,

        50,

        100,

        200,

        500,

        1000,

        2000,

        5000,

        10000,

        20000,

        50000,
    ]


    report_min = (
        df["pair_report_count"].min()
    )


    report_max = (
        df["pair_report_count"].max()
    )


    report_ticks = [

        value

        for value in possible_report_ticks

        if report_min <= value <= report_max
    ]


    if report_ticks:

        ax.set_xticks(

            np.log10(report_ticks)
        )


        ax.set_xticklabels(

            [

                f"{value:,}"

                for value in report_ticks

            ]
        )


    # ========================================================
    # TITLE / AXES
    # ========================================================

    ax.set_title(

        "Preliminary Drug–Event Signal Prioritization Map",

        pad=20,

        fontweight="bold"
    )


    ax.set_xlabel(

        "Reporting Support — Number of FAERS Reports (log scale)",

        labelpad=12
    )


    ax.set_ylabel(

        "Disproportionality Strength — IC",

        labelpad=12
    )


    # ========================================================
    # LEGEND
    # ========================================================

    ax.legend(

        title="Review category",

        loc="best",

        frameon=True
    )


    # ========================================================
    # FOOTNOTE
    # ========================================================

    fig.text(

        0.5,

        0.015,

        (
            "Priority indicates relative priority for further "
            "pharmacovigilance review, not causality or clinical risk. "
            "Quadrants are defined using the median report count and "
            "median IC among the 50 displayed associations."
        ),

        ha="center",

        fontsize=10
    )


    # ========================================================
    # STYLE
    # ========================================================

    ax.grid(

        alpha=0.12,

        linewidth=0.7
    )


    ax.spines["top"].set_visible(
        False
    )


    ax.spines["right"].set_visible(
        False
    )


    fig.tight_layout(

        rect=[
            0,
            0.04,
            1,
            1
        ]
    )


    # ========================================================
    # SAVE
    # ========================================================

    fig.savefig(

        OUTPUT_PNG,

        dpi=PNG_DPI,

        bbox_inches="tight",

        facecolor="white"
    )


    fig.savefig(

        OUTPUT_PDF,

        bbox_inches="tight",

        facecolor="white"
    )


    plt.close(fig)


    print(
        f"\nCreated:\n{OUTPUT_PNG}"
    )


    print(
        f"\nCreated:\n{OUTPUT_PDF}"
    )


# ============================================================
# SAVE TABLES
# ============================================================

def save_tables(df):

    print("\n" + "=" * 78)
    print("CREATING PRIORITY TABLES")
    print("=" * 78)


    output_columns = [

        "review_priority_rank",

        "original_rank",

        "suspect_drug",

        "adverse_event",

        "priority_region",

        "pair_report_count",

        "ror",

        "ror_ci_lower",

        "ror_ci_upper",

        "prr",

        "ic",
    ]


    top50_table = df[
        output_columns
    ].copy()


    # --------------------------------------------------------
    # Round values for readability
    # --------------------------------------------------------

    for column in [

        "ror",

        "ror_ci_lower",

        "ror_ci_upper",

        "prr",

        "ic",
    ]:

        top50_table[column] = (
            top50_table[column]
            .round(2)
        )


    # ========================================================
    # SAVE TOP 50
    # ========================================================

    top50_table.to_csv(

        OUTPUT_TOP50,

        index=False
    )


    # ========================================================
    # SAVE TOP 10 FOR POSTER
    # ========================================================

    top10_table = (
        top50_table
        .head(10)
        .copy()
    )


    top10_table.to_csv(

        OUTPUT_TOP10,

        index=False
    )


    print(
        f"\nCreated:\n{OUTPUT_TOP50}"
    )


    print(
        f"\nCreated:\n{OUTPUT_TOP10}"
    )


    # ========================================================
    # PRINT TOP 10
    # ========================================================

    print("\n" + "-" * 78)

    print(
        "TOP 10 ASSOCIATIONS FOR FURTHER REVIEW"
    )

    print("-" * 78)


    display_columns = [

        "review_priority_rank",

        "suspect_drug",

        "adverse_event",

        "pair_report_count",

        "ror",

        "prr",

        "ic",

        "priority_region",
    ]


    print(

        top10_table[
            display_columns
        ].to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load first 50 supported associations
    # --------------------------------------------------------

    df = load_data()


    # --------------------------------------------------------
    # Calculate priority variables
    # --------------------------------------------------------

    (
        priority_df,

        report_boundary,

        ic_boundary,

        log_report_boundary

    ) = build_priority_variables(df)


    # --------------------------------------------------------
    # Create graph
    # --------------------------------------------------------

    create_priority_matrix(

        priority_df,

        report_boundary,

        ic_boundary,

        log_report_boundary
    )


    # --------------------------------------------------------
    # Create tables
    # --------------------------------------------------------

    save_tables(
        priority_df
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 78)

    print(
        "PRIORITY MATRIX COMPLETE"
    )

    print("=" * 78)


    print(
        "\nFILES CREATED BY THIS SCRIPT:"
    )


    print(
        f"\n1. {OUTPUT_PNG}"
    )


    print(
        f"\n2. {OUTPUT_PDF}"
    )


    print(
        f"\n3. {OUTPUT_TOP50}"
    )


    print(
        f"\n4. {OUTPUT_TOP10}"
    )


    print(
        "\nRECOMMENDED POSTER GRAPH:"
    )


    print(
        OUTPUT_PNG
    )


    print(
        "\nRECOMMENDED POSTER TABLE:"
    )


    print(
        OUTPUT_TOP10
    )


    print(
        "\nIMPORTANT:"
    )


    print(
        "Priority means priority for further pharmacovigilance "
        "review — not confirmed causality or clinical risk."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print("\n" + "=" * 78)

        print(
            "PRIORITY MATRIX FAILED"
        )

        print("=" * 78)


        print(
            f"\n{type(error).__name__}: "
            f"{error}"
        )


        raise