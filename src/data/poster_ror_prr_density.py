from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PHARMAAI — POSTER ROR vs PRR DENSITY VISUALIZATION
# ============================================================
#
# PURPOSE:
#
# Create a poster-quality visualization showing the relationship
# between Reporting Odds Ratio (ROR) and Proportional Reporting
# Ratio (PRR) across supported FAERS drug-event associations.
#
#
# INPUT FILE:
#
# PharmaAI/
# └── results/
#     └── preliminary/
#         └── tables/
#             └── signals_min_10_reports.csv
#
#
# FILES CREATED BY THIS SCRIPT:
#
# PharmaAI/
# └── results/
#     └── preliminary/
#         └── poster_graphs/
#
#             1. 05_ror_vs_prr_hexbin.png
#                - 600-DPI poster-ready density graph
#                - RECOMMENDED PNG FOR POSTER
#
#             2. 05_ror_vs_prr_hexbin.pdf
#                - Vector PDF version
#                - Recommended if poster software supports PDF
#
#             3. 05_ror_vs_prr_transparent_scatter.png
#                - Alternative transparent scatter visualization
#
#             4. 05_ror_vs_prr_transparent_scatter.pdf
#                - Vector/rasterized PDF version of scatter plot
#
#
# IMPORTANT:
#
# - This script DOES NOT modify FAERS data.
# - This script DOES NOT modify signal tables.
# - This script DOES NOT recalculate ROR or PRR.
# - It only visualizes previously calculated results.
#
# - Only associations with >= 10 reports are displayed.
# - ROR and PRR are displayed on logarithmic scales.
# - Dashed lines indicate:
#
#       ROR = 2
#       PRR = 2
#
# - Extreme values are trimmed ONLY FOR VISUALIZATION.
#   They remain unchanged in the original CSV.
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

POSTER_GRAPH_DIR = (
    PROJECT_ROOT
    / "results"
    / "preliminary"
    / "poster_graphs"
)

POSTER_GRAPH_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# INPUT FILE
# ============================================================

INPUT_FILE = (
    TABLE_DIR
    / "signals_min_10_reports.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

HEXBIN_PNG = (
    POSTER_GRAPH_DIR
    / "05_ror_vs_prr_hexbin.png"
)

HEXBIN_PDF = (
    POSTER_GRAPH_DIR
    / "05_ror_vs_prr_hexbin.pdf"
)

SCATTER_PNG = (
    POSTER_GRAPH_DIR
    / "05_ror_vs_prr_transparent_scatter.png"
)

SCATTER_PDF = (
    POSTER_GRAPH_DIR
    / "05_ror_vs_prr_transparent_scatter.pdf"
)


# ============================================================
# ANALYSIS / VISUALIZATION SETTINGS
# ============================================================

MIN_PAIR_REPORTS = 10

ROR_THRESHOLD = 2.0

PRR_THRESHOLD = 2.0


# Number of hexagonal bins.
#
# Larger value:
#     more detailed density structure
#
# Smaller value:
#     smoother density structure
#
# 90 is a good poster-quality starting point.
# ============================================================

HEXBIN_GRIDSIZE = 90


# ============================================================
# DISPLAY TRIMMING
# ============================================================
#
# Very extreme ROR/PRR values can stretch the plot and make
# the majority of associations difficult to see.
#
# We therefore display approximately the 0.1th–99.5th
# percentile range.
#
# IMPORTANT:
# This DOES NOT delete observations from your source data.
# It affects only this visualization.
# ============================================================

LOWER_DISPLAY_PERCENTILE = 0.1

UPPER_DISPLAY_PERCENTILE = 99.5


# ============================================================
# POSTER FIGURE SETTINGS
# ============================================================

plt.rcParams.update({

    "figure.dpi": 150,

    "savefig.dpi": 600,

    "font.size": 15,

    "axes.titlesize": 21,

    "axes.labelsize": 17,

    "xtick.labelsize": 13,

    "ytick.labelsize": 13,

    "legend.fontsize": 12,

    "figure.titlesize": 22,
})


# ============================================================
# LOAD SIGNAL DATA
# ============================================================

def load_signal_data():

    print("\n" + "=" * 78)

    print(
        "PHARMAAI — ROR vs PRR POSTER DENSITY VISUALIZATION"
    )

    print("=" * 78)

    print(
        f"\nInput file:\n{INPUT_FILE}"
    )


    # --------------------------------------------------------
    # Check input file
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(

            f"\nCould not find the required input file:\n"
            f"{INPUT_FILE}\n\n"

            f"Run preliminary_signal_analysis_revised.py "
            f"before running this visualization script."
        )


    # --------------------------------------------------------
    # Load only required columns
    # --------------------------------------------------------

    required_columns = [

        "suspect_drug",

        "adverse_event",

        "pair_report_count",

        "ror",

        "prr",
    ]


    df = pd.read_csv(

        INPUT_FILE,

        usecols=required_columns
    )


    print(
        f"\nRows loaded: {len(df):,}"
    )


    return df


# ============================================================
# PREPARE VISUALIZATION DATA
# ============================================================

def prepare_plot_data(df):

    print("\n" + "=" * 78)

    print(
        "PREPARING VISUALIZATION DATA"
    )

    print("=" * 78)


    # --------------------------------------------------------
    # Convert numerical columns
    # --------------------------------------------------------

    numeric_columns = [

        "pair_report_count",

        "ror",

        "prr",
    ]


    for column in numeric_columns:

        df[column] = pd.to_numeric(

            df[column],

            errors="coerce"
        )


    # --------------------------------------------------------
    # Remove missing numerical values
    # --------------------------------------------------------

    before = len(df)


    df = df.dropna(

        subset=[

            "pair_report_count",

            "ror",

            "prr",
        ]

    ).copy()


    print(
        f"\nRows removed because of missing numerical values: "
        f"{before - len(df):,}"
    )


    # --------------------------------------------------------
    # Apply support requirement
    # --------------------------------------------------------

    before = len(df)


    df = df[

        df["pair_report_count"]

        >= MIN_PAIR_REPORTS

    ].copy()


    print(
        f"Rows removed below {MIN_PAIR_REPORTS} reports: "
        f"{before - len(df):,}"
    )


    # --------------------------------------------------------
    # Logarithmic visualization requires positive values
    # --------------------------------------------------------

    before = len(df)


    df = df[

        (df["ror"] > 0)

        &

        (df["prr"] > 0)

    ].copy()


    print(
        f"Rows removed because ROR/PRR were non-positive: "
        f"{before - len(df):,}"
    )


    print(
        f"\nSupported associations available: "
        f"{len(df):,}"
    )


    if df.empty:

        raise ValueError(

            "No valid associations remain after filtering."
        )


    # ========================================================
    # DETERMINE DISPLAY LIMITS
    # ========================================================

    ror_lower = np.nanpercentile(

        df["ror"],

        LOWER_DISPLAY_PERCENTILE
    )


    ror_upper = np.nanpercentile(

        df["ror"],

        UPPER_DISPLAY_PERCENTILE
    )


    prr_lower = np.nanpercentile(

        df["prr"],

        LOWER_DISPLAY_PERCENTILE
    )


    prr_upper = np.nanpercentile(

        df["prr"],

        UPPER_DISPLAY_PERCENTILE
    )


    # --------------------------------------------------------
    # Ensure lower values remain positive
    # --------------------------------------------------------

    smallest_positive_ror = (

        df.loc[

            df["ror"] > 0,

            "ror"

        ].min()
    )


    smallest_positive_prr = (

        df.loc[

            df["prr"] > 0,

            "prr"

        ].min()
    )


    ror_lower = max(

        ror_lower,

        smallest_positive_ror
    )


    prr_lower = max(

        prr_lower,

        smallest_positive_prr
    )


    # ========================================================
    # CREATE DISPLAY DATASET
    # ========================================================

    plot_df = df[

        (df["ror"] >= ror_lower)

        &

        (df["ror"] <= ror_upper)

        &

        (df["prr"] >= prr_lower)

        &

        (df["prr"] <= prr_upper)

    ].copy()


    print(
        f"\nAssociations displayed after visualization trimming: "
        f"{len(plot_df):,}"
    )


    print(
        "\nDISPLAY RANGE"
    )


    print(
        f"\nROR:"
        f"\n  Minimum = {ror_lower:.4f}"
        f"\n  Maximum = {ror_upper:.4f}"
    )


    print(
        f"\nPRR:"
        f"\n  Minimum = {prr_lower:.4f}"
        f"\n  Maximum = {prr_upper:.4f}"
    )


    # ========================================================
    # LOG10 TRANSFORMATION
    # ========================================================
    #
    # We manually transform the coordinates because this gives
    # us reliable hexagonal binning while still allowing the
    # graph labels to show the original ROR/PRR values.
    # ========================================================

    plot_df["log10_ror"] = np.log10(

        plot_df["ror"]
    )


    plot_df["log10_prr"] = np.log10(

        plot_df["prr"]
    )


    return plot_df


# ============================================================
# CREATE LOG TICKS
# ============================================================

def create_log_ticks(

    minimum,

    maximum
):

    possible_ticks = [

        0.001,

        0.01,

        0.1,

        1,

        2,

        10,

        100,

        1000,

        10000,

        100000,
    ]


    ticks = [

        value

        for value in possible_ticks

        if minimum <= value <= maximum
    ]


    return ticks


# ============================================================
# HEXBIN DENSITY PLOT
# ============================================================

def create_hexbin_plot(df):

    print("\n" + "=" * 78)

    print(
        "CREATING POSTER HEXBIN DENSITY GRAPH"
    )

    print("=" * 78)


    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(

        figsize=(13, 9)
    )


    # ========================================================
    # HEXBIN
    # ========================================================
    #
    # Each hexagon represents a region of ROR/PRR space.
    #
    # More associations in a region produce greater density.
    #
    # bins="log" prevents very dense regions from overwhelming
    # lower-density regions.
    # ========================================================

    hexbin = ax.hexbin(

        df["log10_ror"],

        df["log10_prr"],

        gridsize=HEXBIN_GRIDSIZE,

        bins="log",

        mincnt=1,

        linewidths=0.10
    )


    # ========================================================
    # SCREENING THRESHOLDS
    # ========================================================

    ror_threshold_log = np.log10(

        ROR_THRESHOLD
    )


    prr_threshold_log = np.log10(

        PRR_THRESHOLD
    )


    ax.axvline(

        ror_threshold_log,

        linestyle="--",

        linewidth=1.5,

        label=f"ROR = {ROR_THRESHOLD:g}"
    )


    ax.axhline(

        prr_threshold_log,

        linestyle="--",

        linewidth=1.5,

        label=f"PRR = {PRR_THRESHOLD:g}"
    )


    # ========================================================
    # X AXIS TICKS
    # ========================================================

    x_ticks = create_log_ticks(

        df["ror"].min(),

        df["ror"].max()
    )


    ax.set_xticks(

        np.log10(x_ticks)
    )


    ax.set_xticklabels(

        [

            f"{value:g}"

            for value in x_ticks

        ]
    )


    # ========================================================
    # Y AXIS TICKS
    # ========================================================

    y_ticks = create_log_ticks(

        df["prr"].min(),

        df["prr"].max()
    )


    ax.set_yticks(

        np.log10(y_ticks)
    )


    ax.set_yticklabels(

        [

            f"{value:g}"

            for value in y_ticks

        ]
    )


    # ========================================================
    # TITLE
    # ========================================================

    ax.set_title(

        "Density of ROR and PRR Across FAERS Drug–Event Associations",

        pad=18,

        fontweight="bold"
    )


    # ========================================================
    # AXIS LABELS
    # ========================================================

    ax.set_xlabel(

        "Reporting Odds Ratio (ROR, log scale)",

        labelpad=10
    )


    ax.set_ylabel(

        "Proportional Reporting Ratio (PRR, log scale)",

        labelpad=10
    )


    # ========================================================
    # COLOR BAR
    # ========================================================

    colorbar = fig.colorbar(

        hexbin,

        ax=ax,

        pad=0.02
    )


    colorbar.set_label(

        "Association density",

        fontsize=15
    )


    colorbar.ax.tick_params(

        labelsize=12
    )


    # ========================================================
    # LEGEND
    # ========================================================

    ax.legend(

        loc="upper left",

        frameon=True
    )


    # ========================================================
    # POSTER FORMATTING
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


    # ========================================================
    # SUPPORT NOTE
    # ========================================================

    ax.text(

        0.99,

        0.02,

        f"Associations with ≥{MIN_PAIR_REPORTS} reports",

        transform=ax.transAxes,

        horizontalalignment="right",

        verticalalignment="bottom",

        fontsize=11
    )


    fig.tight_layout()


    # ========================================================
    # SAVE PNG
    # ========================================================

    fig.savefig(

        HEXBIN_PNG,

        dpi=600,

        bbox_inches="tight",

        facecolor="white"
    )


    # ========================================================
    # SAVE PDF
    # ========================================================

    fig.savefig(

        HEXBIN_PDF,

        bbox_inches="tight",

        facecolor="white"
    )


    plt.close(fig)


    print(
        f"\nCreated:\n{HEXBIN_PNG}"
    )


    print(
        f"\nCreated:\n{HEXBIN_PDF}"
    )


# ============================================================
# TRANSPARENT SCATTER VERSION
# ============================================================

def create_scatter_plot(df):

    print("\n" + "=" * 78)

    print(
        "CREATING ALTERNATIVE TRANSPARENT SCATTER GRAPH"
    )

    print("=" * 78)


    fig, ax = plt.subplots(

        figsize=(13, 9)
    )


    # ========================================================
    # SCATTER
    # ========================================================
    #
    # Small points + low alpha reduce overplotting.
    #
    # rasterized=True keeps the PDF manageable even with
    # hundreds of thousands of points.
    # ========================================================

    ax.scatter(

        df["log10_ror"],

        df["log10_prr"],

        s=4,

        alpha=0.04,

        rasterized=True
    )


    # ========================================================
    # THRESHOLD LINES
    # ========================================================

    ax.axvline(

        np.log10(ROR_THRESHOLD),

        linestyle="--",

        linewidth=1.5,

        label=f"ROR = {ROR_THRESHOLD:g}"
    )


    ax.axhline(

        np.log10(PRR_THRESHOLD),

        linestyle="--",

        linewidth=1.5,

        label=f"PRR = {PRR_THRESHOLD:g}"
    )


    # ========================================================
    # AXIS TICKS
    # ========================================================

    x_ticks = create_log_ticks(

        df["ror"].min(),

        df["ror"].max()
    )


    y_ticks = create_log_ticks(

        df["prr"].min(),

        df["prr"].max()
    )


    ax.set_xticks(

        np.log10(x_ticks)
    )


    ax.set_xticklabels(

        [

            f"{value:g}"

            for value in x_ticks

        ]
    )


    ax.set_yticks(

        np.log10(y_ticks)
    )


    ax.set_yticklabels(

        [

            f"{value:g}"

            for value in y_ticks

        ]
    )


    # ========================================================
    # LABELS
    # ========================================================

    ax.set_title(

        "ROR vs PRR Across Supported FAERS Drug–Event Associations",

        pad=18,

        fontweight="bold"
    )


    ax.set_xlabel(

        "Reporting Odds Ratio (ROR, log scale)",

        labelpad=10
    )


    ax.set_ylabel(

        "Proportional Reporting Ratio (PRR, log scale)",

        labelpad=10
    )


    # ========================================================
    # LEGEND
    # ========================================================

    ax.legend(

        loc="upper left",

        frameon=True
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


    ax.text(

        0.99,

        0.02,

        f"Associations with ≥{MIN_PAIR_REPORTS} reports",

        transform=ax.transAxes,

        horizontalalignment="right",

        verticalalignment="bottom",

        fontsize=11
    )


    fig.tight_layout()


    # ========================================================
    # SAVE
    # ========================================================

    fig.savefig(

        SCATTER_PNG,

        dpi=600,

        bbox_inches="tight",

        facecolor="white"
    )


    fig.savefig(

        SCATTER_PDF,

        bbox_inches="tight",

        facecolor="white"
    )


    plt.close(fig)


    print(
        f"\nCreated:\n{SCATTER_PNG}"
    )


    print(
        f"\nCreated:\n{SCATTER_PDF}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load existing signal calculations
    # --------------------------------------------------------

    df = load_signal_data()


    # --------------------------------------------------------
    # Prepare data for visualization
    # --------------------------------------------------------

    plot_df = prepare_plot_data(

        df
    )


    # --------------------------------------------------------
    # Recommended density graph
    # --------------------------------------------------------

    create_hexbin_plot(

        plot_df
    )


    # --------------------------------------------------------
    # Alternative scatter graph
    # --------------------------------------------------------

    create_scatter_plot(

        plot_df
    )


    # ========================================================
    # FINAL OUTPUT SUMMARY
    # ========================================================

    print("\n" + "=" * 78)

    print(
        "POSTER VISUALIZATION COMPLETE"
    )

    print("=" * 78)


    print(
        "\nINPUT FILE:"
    )


    print(

        INPUT_FILE
    )


    print(
        "\nFILES CREATED BY THIS SCRIPT:"
    )


    print(
        f"\n1. {HEXBIN_PNG.name}"
    )

    print(
        f"   {HEXBIN_PNG}"
    )

    print(
        "\n   Purpose:"
        "\n   600-DPI hexbin density visualization."
    )


    print(
        f"\n2. {HEXBIN_PDF.name}"
    )

    print(
        f"   {HEXBIN_PDF}"
    )

    print(
        "\n   Purpose:"
        "\n   Vector PDF version of the hexbin visualization."
    )


    print(
        f"\n3. {SCATTER_PNG.name}"
    )

    print(
        f"   {SCATTER_PNG}"
    )

    print(
        "\n   Purpose:"
        "\n   Alternative transparent scatter visualization."
    )


    print(
        f"\n4. {SCATTER_PDF.name}"
    )

    print(
        f"   {SCATTER_PDF}"
    )

    print(
        "\n   Purpose:"
        "\n   PDF version of the transparent scatter visualization."
    )


    print("\n" + "-" * 78)

    print(
        "RECOMMENDED POSTER GRAPH:"
    )

    print("-" * 78)


    print(
        f"\n{HEXBIN_PNG}"
    )


    print(
        "\nIf your poster software supports PDF graphics, "
        "use the vector PDF:"
    )


    print(
        f"\n{HEXBIN_PDF}"
    )


    print("\n" + "-" * 78)

    print(
        "IMPORTANT:"
    )

    print("-" * 78)


    print(
        "\nNo FAERS source data were modified."
    )


    print(
        "No ROR/PRR signal calculations were modified."
    )


    print(
        "Display trimming affects only the visualization."
    )


    print(
        "ROR/PRR indicate disproportional reporting, "
        "not causal drug-event relationships."
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
            "VISUALIZATION FAILED"
        )

        print("=" * 78)


        print(
            f"\n{type(error).__name__}: "
            f"{error}"
        )


        raise