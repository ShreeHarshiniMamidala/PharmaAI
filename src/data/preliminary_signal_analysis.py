from pathlib import Path
import math
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PHARMAAI - PRELIMINARY SIGNAL ANALYSIS
# ============================================================
#
# Purpose:
#   1. Read processed FAERS dashboard Parquet files
#   2. Build suspect-drug x adverse-event pairs
#   3. Calculate descriptive statistics
#   4. Calculate ROR + 95% CI and PRR
#   5. Rank preliminary candidate signals
#   6. Save poster-ready tables and high-resolution figures
#
# IMPORTANT:
#   - These are preliminary disproportionality results.
#   - ROR/PRR identify reporting disproportionality, NOT causality.
#   - This script does not yet implement IC or ML models.
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\mamid\OneDrive\Desktop\PharmaAI"
)

DASHBOARD_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dashboard"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "preliminary"
)

TABLE_DIR = RESULTS_DIR / "tables"
POSTER_GRAPH_DIR = RESULTS_DIR / "poster_graphs"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
POSTER_GRAPH_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Quarters to analyze
# ------------------------------------------------------------

PROCESS_QUARTERS = {
    "2025_Q1",
    "2025_Q2",
    "2025_Q3",
    "2025_Q4",
}


# ------------------------------------------------------------
# Signal filtering
# ------------------------------------------------------------
#
# A minimum count reduces unstable rankings caused by extremely
# rare drug-event pairs. For a preliminary poster, 5 is a
# reasonable transparent starting threshold.
# ------------------------------------------------------------

MIN_PAIR_REPORTS = 10
MIN_ROR = 2.0
MIN_PRR = 2.0
MIN_IC = 1.0

TOP_N = 10

# Prevent a few extreme values from making the ROR-vs-PRR
# scatter plot unreadable. This affects the visualization only,
# NOT the saved signal calculations/rankings.
SCATTER_PERCENTILE_LIMIT = 99


# ============================================================
# POSTER FIGURE SETTINGS
# ============================================================

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "font.size": 14,
    "axes.titlesize": 20,
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 22,
})


# ============================================================
# HELPERS
# ============================================================

def normalize_text(value):
    """
    Normalize a drug or adverse-event term for analysis.

    We preserve meaningful punctuation but:
      - strip whitespace
      - collapse repeated whitespace
      - convert to uppercase

    Returns None for missing/empty values.
    """

    if value is None:
        return None

    if pd.isna(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    value = re.sub(r"\s+", " ", value)

    return value.upper()


def split_pipe_values(value):
    """
    Convert the extractor's pipe-separated values into a
    de-duplicated list.
    """

    if value is None or pd.isna(value):
        return []

    values = []

    seen = set()

    for item in str(value).split("|"):

        item = normalize_text(item)

        if item is None:
            continue

        if item not in seen:
            values.append(item)
            seen.add(item)

    return values


def save_figure(fig, filename_stem):
    """
    Save each poster graph as:
      - 600-DPI PNG
      - vector PDF
    """

    png_path = POSTER_GRAPH_DIR / f"{filename_stem}.png"
    pdf_path = POSTER_GRAPH_DIR / f"{filename_stem}.pdf"

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
        facecolor="white"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)

    print(f"Saved graph: {png_path}")
    print(f"Saved graph: {pdf_path}")


def clean_axis(ax):
    """
    Light poster-friendly formatting.
    """

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        axis="x",
        alpha=0.20
    )


def make_pair_label(drug, event, max_chars=75):
    label = f"{drug} — {event}"

    if len(label) > max_chars:
        return label[:max_chars - 1] + "…"

    return label


# ============================================================
# LOAD DASHBOARD DATA
# ============================================================

def load_dashboard_data():

    print("\n" + "=" * 78)
    print("LOADING FAERS DASHBOARD DATA")
    print("=" * 78)

    if not DASHBOARD_DIR.exists():
        raise FileNotFoundError(
            f"Dashboard directory does not exist:\n{DASHBOARD_DIR}"
        )

    parquet_files = []

    for quarter in sorted(PROCESS_QUARTERS):

        quarter_dir = DASHBOARD_DIR / quarter

        if not quarter_dir.exists():
            print(f"WARNING: Missing quarter directory: {quarter_dir}")
            continue

        files = sorted(
            quarter_dir.glob("*.parquet")
        )

        print(
            f"{quarter}: "
            f"{len(files)} Parquet file(s)"
        )

        parquet_files.extend(files)

    if not parquet_files:
        raise FileNotFoundError(
            "No dashboard Parquet files were found for the selected quarters."
        )

    frames = []

    required_columns = [
        "safetyreportid",
        "year",
        "quarter",
        "serious",
        "patient_age_years",
        "patient_sex",
        "suspect_drug_names",
        "reaction_terms",
    ]

    for number, file_path in enumerate(
        parquet_files,
        start=1
    ):

        print(
            f"[{number}/{len(parquet_files)}] "
            f"Reading {file_path.name}"
        )

        df = pd.read_parquet(
            file_path
        )

        missing = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"\nRequired column(s) missing from:\n"
                f"{file_path}\n"
                f"Missing: {missing}\n\n"
                f"Make sure this is a dashboard Parquet created "
                f"by your current extract_faers.py."
            )

        frames.append(
            df[required_columns].copy()
        )

    dashboard = pd.concat(
        frames,
        ignore_index=True
    )

    # Safety report IDs are the unit used for contingency counts.
    dashboard["safetyreportid"] = (
        dashboard["safetyreportid"]
        .astype("string")
    )

    missing_id_count = (
        dashboard["safetyreportid"]
        .isna()
        .sum()
    )

    if missing_id_count:
        print(
            f"WARNING: Dropping {missing_id_count:,} "
            f"rows with missing safetyreportid."
        )

        dashboard = dashboard[
            dashboard["safetyreportid"].notna()
        ].copy()

    print(
        f"\nRows loaded: "
        f"{len(dashboard):,}"
    )

    print(
        f"Unique safety reports: "
        f"{dashboard['safetyreportid'].nunique():,}"
    )

    return dashboard


# ============================================================
# BUILD DRUG-EVENT PAIRS
# ============================================================

def build_drug_event_pairs(dashboard):
    """
    Build report-level suspect-drug x adverse-event combinations.

    Example:
        Report:
          suspect drugs = A | B
          reactions     = X | Y

        Pairs:
          A-X
          A-Y
          B-X
          B-Y

    Each report contributes at most once to a specific pair.
    """

    print("\n" + "=" * 78)
    print("BUILDING SUSPECT DRUG x ADVERSE-EVENT PAIRS")
    print("=" * 78)

    rows = []

    usable_reports = 0
    no_suspect_drug = 0
    no_reaction = 0

    columns = [
        "safetyreportid",
        "year",
        "quarter",
        "serious",
        "patient_age_years",
        "patient_sex",
        "suspect_drug_names",
        "reaction_terms",
    ]

    for (
        safetyreportid,
        year,
        quarter,
        serious,
        patient_age_years,
        patient_sex,
        suspect_drug_names,
        reaction_terms,
    ) in dashboard[columns].itertuples(
        index=False,
        name=None
    ):

        drugs = split_pipe_values(
            suspect_drug_names
        )

        events = split_pipe_values(
            reaction_terms
        )

        if not drugs:
            no_suspect_drug += 1
            continue

        if not events:
            no_reaction += 1
            continue

        usable_reports += 1

        # Avoid duplicate pair contribution inside the same report.
        pair_set = {
            (drug, event)
            for drug in drugs
            for event in events
        }

        for drug, event in pair_set:

            rows.append({
                "safetyreportid": safetyreportid,
                "year": year,
                "quarter": quarter,
                "suspect_drug": drug,
                "adverse_event": event,
                "serious": serious,
                "patient_age_years": patient_age_years,
                "patient_sex": patient_sex,
            })

    pairs = pd.DataFrame(rows)

    if pairs.empty:
        raise ValueError(
            "No suspect-drug/adverse-event pairs could be constructed."
        )

    pairs = (
        pairs
        .drop_duplicates(
            subset=[
                "safetyreportid",
                "suspect_drug",
                "adverse_event",
            ]
        )
        .reset_index(drop=True)
    )

    print(
        f"Reports with usable drug-event data: "
        f"{usable_reports:,}"
    )

    print(
        f"Reports skipped - no suspect drug: "
        f"{no_suspect_drug:,}"
    )

    print(
        f"Reports skipped - no reaction: "
        f"{no_reaction:,}"
    )

    print(
        f"Report-level drug-event pair rows: "
        f"{len(pairs):,}"
    )

    print(
        f"Unique suspect drugs: "
        f"{pairs['suspect_drug'].nunique():,}"
    )

    print(
        f"Unique adverse events: "
        f"{pairs['adverse_event'].nunique():,}"
    )

    print(
        f"Unique drug-event combinations: "
        f"{pairs[['suspect_drug', 'adverse_event']].drop_duplicates().shape[0]:,}"
    )

    return pairs


# ============================================================
# DATASET SUMMARY
# ============================================================

def create_dataset_summary(
    dashboard,
    pairs
):

    print("\n" + "=" * 78)
    print("CREATING DATASET SUMMARY")
    print("=" * 78)

    total_reports = (
        dashboard["safetyreportid"]
        .nunique()
    )

    usable_pair_reports = (
        pairs["safetyreportid"]
        .nunique()
    )

    unique_drugs = (
        pairs["suspect_drug"]
        .nunique()
    )

    unique_events = (
        pairs["adverse_event"]
        .nunique()
    )

    unique_pairs = (
        pairs[
            [
                "suspect_drug",
                "adverse_event"
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    # FAERS: serious 1 = serious, 2 = non-serious
    serious_numeric = pd.to_numeric(
        dashboard["serious"],
        errors="coerce"
    )

    serious_percent = (
        serious_numeric
        .eq(1)
        .mean()
        * 100
    )

    age_numeric = pd.to_numeric(
        dashboard["patient_age_years"],
        errors="coerce"
    )

    median_age = (
        age_numeric
        .median()
    )

    summary = pd.DataFrame({
        "metric": [
            "FAERS period",
            "Total safety reports",
            "Reports usable for drug-event analysis",
            "Unique suspect drugs",
            "Unique adverse events",
            "Unique suspect-drug/adverse-event pairs",
            "Serious reports (%)",
            "Median patient age (years)",
        ],
        "value": [
            ", ".join(sorted(PROCESS_QUARTERS)),
            total_reports,
            usable_pair_reports,
            unique_drugs,
            unique_events,
            unique_pairs,
            round(serious_percent, 2),
            (
                round(float(median_age), 2)
                if pd.notna(median_age)
                else np.nan
            ),
        ]
    })

    output = (
        TABLE_DIR
        / "dataset_summary.csv"
    )

    summary.to_csv(
        output,
        index=False
    )

    print(summary.to_string(index=False))
    print(f"\nSaved: {output}")

    return summary


# ============================================================
# FREQUENCY TABLES
# ============================================================

def create_frequency_tables(pairs):

    print("\n" + "=" * 78)
    print("CREATING FREQUENCY TABLES")
    print("=" * 78)

    drug_counts = (
        pairs[
            [
                "safetyreportid",
                "suspect_drug"
            ]
        ]
        .drop_duplicates()
        .groupby(
            "suspect_drug"
        )["safetyreportid"]
        .nunique()
        .sort_values(
            ascending=False
        )
        .rename("report_count")
        .reset_index()
    )

    event_counts = (
        pairs[
            [
                "safetyreportid",
                "adverse_event"
            ]
        ]
        .drop_duplicates()
        .groupby(
            "adverse_event"
        )["safetyreportid"]
        .nunique()
        .sort_values(
            ascending=False
        )
        .rename("report_count")
        .reset_index()
    )

    pair_counts = (
        pairs
        .groupby(
            [
                "suspect_drug",
                "adverse_event"
            ]
        )["safetyreportid"]
        .nunique()
        .sort_values(
            ascending=False
        )
        .rename("pair_report_count")
        .reset_index()
    )

    drug_counts.to_csv(
        TABLE_DIR / "suspect_drug_frequencies.csv",
        index=False
    )

    event_counts.to_csv(
        TABLE_DIR / "adverse_event_frequencies.csv",
        index=False
    )

    pair_counts.to_csv(
        TABLE_DIR / "drug_event_pair_frequencies.csv",
        index=False
    )

    return (
        drug_counts,
        event_counts,
        pair_counts
    )


# ============================================================
# ROR / PRR
# ============================================================

def calculate_signals(
    pairs,
    pair_counts
):
    """
    Calculate report-based disproportionality statistics.

    For drug D and event E:

                        Event E    Other events
        Drug D             a            b
        Other drugs        c            d

    a = reports containing D and E
    b = reports containing D but not E
    c = reports containing E but not D
    d = reports containing neither D nor E

    ROR = (a*d)/(b*c)

    PRR = [a/(a+b)] / [c/(c+d)]

    ROR 95% CI is calculated on the log scale:
        log(ROR) +/- 1.96 * sqrt(1/a + 1/b + 1/c + 1/d)
    """

    print("\n" + "=" * 78)
    print("CALCULATING ROR, 95% CI, AND PRR")
    print("=" * 78)

    report_universe = (
        pairs["safetyreportid"]
        .nunique()
    )

    drug_report_counts = (
        pairs[
            [
                "safetyreportid",
                "suspect_drug"
            ]
        ]
        .drop_duplicates()
        .groupby(
            "suspect_drug"
        )["safetyreportid"]
        .nunique()
        .to_dict()
    )

    event_report_counts = (
        pairs[
            [
                "safetyreportid",
                "adverse_event"
            ]
        ]
        .drop_duplicates()
        .groupby(
            "adverse_event"
        )["safetyreportid"]
        .nunique()
        .to_dict()
    )

    signal_rows = []

    for row in pair_counts.itertuples(
        index=False
    ):

        drug = row.suspect_drug
        event = row.adverse_event

        a = int(
            row.pair_report_count
        )

        drug_total = int(
            drug_report_counts[drug]
        )

        event_total = int(
            event_report_counts[event]
        )

        b = drug_total - a
        c = event_total - a
        d = (
            report_universe
            - a
            - b
            - c
        )

        # A valid 2x2 table cannot have negative cells.
        if min(a, b, c, d) < 0:
            continue

        # Haldane-Anscombe correction only when a zero cell exists.
        # This allows finite preliminary estimates while retaining
        # the original integer counts in the output.
        if 0 in (a, b, c, d):

            aa = a + 0.5
            bb = b + 0.5
            cc = c + 0.5
            dd = d + 0.5

            continuity_corrected = True

        else:

            aa = float(a)
            bb = float(b)
            cc = float(c)
            dd = float(d)

            continuity_corrected = False

        ror = (
            (aa * dd)
            /
            (bb * cc)
        )

        se_log_ror = math.sqrt(
            (1 / aa)
            + (1 / bb)
            + (1 / cc)
            + (1 / dd)
        )

        ror_ci_lower = math.exp(
            math.log(ror)
            - 1.96 * se_log_ror
        )

        ror_ci_upper = math.exp(
            math.log(ror)
            + 1.96 * se_log_ror
        )

        drug_event_rate = (
            aa
            /
            (aa + bb)
        )

        background_event_rate = (
            cc
            /
            (cc + dd)
        )

        if background_event_rate > 0:
            prr = (
                drug_event_rate
                /
                background_event_rate
            )
        else:
            prr = np.nan

        expected_count = (
            (drug_total * event_total) / report_universe
            if report_universe > 0 else np.nan
        )
        ic = (
            math.log2(a / expected_count)
            if a > 0 and expected_count > 0 else np.nan
        )

        signal_rows.append({
            "suspect_drug": drug,
            "adverse_event": event,
            "pair_report_count": a,
            "drug_report_count": drug_total,
            "event_report_count": event_total,
            "total_reports_in_signal_universe": report_universe,
            "a_drug_and_event": a,
            "b_drug_other_event": b,
            "c_other_drug_event": c,
            "d_other_drug_other_event": d,
            "ror": ror,
            "ror_ci_lower": ror_ci_lower,
            "ror_ci_upper": ror_ci_upper,
            "prr": prr,
            "expected_pair_count": expected_count,
            "ic": ic,
            "continuity_corrected": continuity_corrected,
        })

    signals = pd.DataFrame(
        signal_rows
    )

    signals = signals.replace(
        [np.inf, -np.inf],
        np.nan
    )

    signals = signals.dropna(
        subset=[
            "ror",
            "ror_ci_lower",
            "ror_ci_upper",
            "prr",
            "ic"
        ]
    )

    signals["meets_min_reports"] = (
        signals["pair_report_count"]
        >= MIN_PAIR_REPORTS
    )

    # Transparent preliminary screening rule.
    # This identifies reporting associations for prioritization,
    # not confirmed causal safety signals.
    signals["meets_ror_screen"] = (
        (signals["ror"] >= MIN_ROR)
        & (signals["ror_ci_lower"] > 1)
    )
    signals["meets_prr_screen"] = signals["prr"] >= MIN_PRR
    signals["meets_ic_screen"] = signals["ic"] >= MIN_IC

    signals["preliminary_candidate_signal"] = (
        signals["meets_min_reports"]
        & signals["meets_ror_screen"]
        & signals["meets_prr_screen"]
        & signals["meets_ic_screen"]
    )

    signals = (
        signals
        .sort_values(
            [
                "preliminary_candidate_signal",
                "pair_report_count",
                "ror_ci_lower",
                "ror"
            ],
            ascending=[
                False,
                False,
                False,
                False
            ]
        )
        .reset_index(drop=True)
    )

    all_output = (
        TABLE_DIR
        / "all_signals.csv"
    )

    signals.to_csv(
        all_output,
        index=False
    )

    filtered = (
        signals[
            signals["meets_min_reports"]
        ]
        .copy()
    )

    filtered_output = (
        TABLE_DIR
        / f"signals_min_{MIN_PAIR_REPORTS}_reports.csv"
    )

    filtered.to_csv(
        filtered_output,
        index=False
    )

    candidate_signals = (
        filtered[
            filtered["preliminary_candidate_signal"]
        ]
        .sort_values(
            [
                "pair_report_count",
                "ror_ci_lower",
                "ror"
            ],
            ascending=False
        )
        .reset_index(drop=True)
    )

    candidate_output = (
        TABLE_DIR
        / "preliminary_candidate_signals.csv"
    )

    candidate_signals.to_csv(
        candidate_output,
        index=False
    )

    top_20 = (
        candidate_signals
        .head(20)
        .copy()
    )

    top_20.to_csv(
        TABLE_DIR / "top_20_supported_signals.csv",
        index=False
    )

    print(
        f"Signal universe reports: "
        f"{report_universe:,}"
    )

    print(
        f"All unique drug-event combinations: "
        f"{len(signals):,}"
    )

    print(
        f"Pairs with >= {MIN_PAIR_REPORTS} reports: "
        f"{len(filtered):,}"
    )

    print(
        f"Preliminary supported candidates "
        f"(reports >= {MIN_PAIR_REPORTS}, ROR >= {MIN_ROR}, "
        f"ROR lower 95% CI > 1, PRR >= {MIN_PRR}, IC >= {MIN_IC}): "
        f"{len(candidate_signals):,}"
    )

    print(f"\nSaved: {all_output}")
    print(f"Saved: {filtered_output}")
    print(f"Saved: {candidate_output}")

    return signals, filtered, candidate_signals


# ============================================================
# POSTER GRAPH 1
# TOP SUSPECT DRUGS
# ============================================================

def plot_top_suspect_drugs(
    drug_counts
):

    data = (
        drug_counts
        .head(TOP_N)
        .sort_values(
            "report_count",
            ascending=True
        )
    )

    fig, ax = plt.subplots(
        figsize=(13, 8)
    )

    ax.barh(
        data["suspect_drug"],
        data["report_count"]
    )

    ax.set_title(
        f"Top {TOP_N} Suspect Drugs in FAERS 2025",
        pad=18,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Number of Safety Reports"
    )

    ax.set_ylabel(
        "Suspect Drug"
    )

    clean_axis(ax)

    fig.tight_layout()

    save_figure(
        fig,
        "01_top_suspect_drugs"
    )


# ============================================================
# POSTER GRAPH 2
# TOP ADVERSE EVENTS
# ============================================================

def plot_top_adverse_events(
    event_counts
):

    data = (
        event_counts
        .head(TOP_N)
        .sort_values(
            "report_count",
            ascending=True
        )
    )

    fig, ax = plt.subplots(
        figsize=(13, 8)
    )

    ax.barh(
        data["adverse_event"],
        data["report_count"]
    )

    ax.set_title(
        f"Top {TOP_N} Reported Adverse Events in FAERS 2025",
        pad=18,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Number of Safety Reports"
    )

    ax.set_ylabel(
        "MedDRA Preferred Term"
    )

    clean_axis(ax)

    fig.tight_layout()

    save_figure(
        fig,
        "02_top_adverse_events"
    )


# ============================================================
# POSTER GRAPH 3
# TOP REPORTED DRUG-EVENT PAIRS
# ============================================================

def plot_top_drug_event_pairs(
    pair_counts
):

    data = (
        pair_counts
        .head(TOP_N)
        .copy()
    )

    data["label"] = [
        make_pair_label(
            drug,
            event
        )
        for drug, event
        in zip(
            data["suspect_drug"],
            data["adverse_event"]
        )
    ]

    data = data.sort_values(
        "pair_report_count",
        ascending=True
    )

    fig, ax = plt.subplots(
        figsize=(15, 9)
    )

    ax.barh(
        data["label"],
        data["pair_report_count"]
    )

    ax.set_title(
        f"Top {TOP_N} Most Frequently Reported Suspect Drug–Event Pairs",
        pad=18,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Number of Safety Reports"
    )

    ax.set_ylabel(
        "Suspect Drug — Adverse Event"
    )

    clean_axis(ax)

    fig.tight_layout()

    save_figure(
        fig,
        "03_top_drug_event_pairs"
    )


# ============================================================
# POSTER GRAPH 4
# ROR + 95% CI
# ============================================================

def plot_top_ror_signals(
    candidate_signals
):

    if candidate_signals.empty:

        print(
            "\nWARNING: No candidate signals meet the "
            "preliminary ROR criterion. "
            "Skipping ROR confidence-interval graph."
        )

        return

    # Ranking by lower CI gives preference to associations whose
    # elevated ROR is more statistically stable.
    data = (
        candidate_signals
        .sort_values(
            [
                "ror_ci_lower",
                "ror"
            ],
            ascending=False
        )
        .head(TOP_N)
        .copy()
    )

    data["label"] = [
        make_pair_label(
            drug,
            event
        )
        for drug, event
        in zip(
            data["suspect_drug"],
            data["adverse_event"]
        )
    ]

    data = data.sort_values(
        "ror",
        ascending=True
    )

    y = np.arange(
        len(data)
    )

    lower_error = (
        data["ror"]
        - data["ror_ci_lower"]
    )

    upper_error = (
        data["ror_ci_upper"]
        - data["ror"]
    )

    fig, ax = plt.subplots(
        figsize=(16, 10)
    )

    ax.errorbar(
        data["ror"],
        y,
        xerr=[
            lower_error,
            upper_error
        ],
        fmt="o",
        capsize=5,
        markersize=7
    )

    ax.axvline(
        1,
        linestyle="--",
        linewidth=1.5
    )

    ax.set_xscale("log")

    ax.set_yticks(y)

    ax.set_yticklabels(
        data["label"]
    )

    ax.set_title(
        "Top Supported Preliminary Drug–Event Reporting Associations",
        pad=18,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Reporting Odds Ratio (ROR), 95% CI — log scale"
    )

    ax.set_ylabel(
        "Suspect Drug — Adverse Event"
    )

    ax.grid(
        axis="x",
        alpha=0.20
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    save_figure(
        fig,
        "04_supported_signals_ror_95ci"
    )


# ============================================================
# POSTER GRAPH 5
# ROR VS PRR
# ============================================================

def plot_ror_vs_prr(
    filtered_signals
):

    data = (
        filtered_signals[
            (
                filtered_signals["ror"] > 0
            )
            &
            (
                filtered_signals["prr"] > 0
            )
        ]
        .copy()
    )

    if data.empty:

        print(
            "\nWARNING: No valid ROR/PRR values available. "
            "Skipping scatter plot."
        )

        return

    # Plotting limits only. Full values remain in CSV outputs.
    ror_limit = np.nanpercentile(
        data["ror"],
        SCATTER_PERCENTILE_LIMIT
    )

    prr_limit = np.nanpercentile(
        data["prr"],
        SCATTER_PERCENTILE_LIMIT
    )

    plot_data = data[
        (
            data["ror"]
            <= ror_limit
        )
        &
        (
            data["prr"]
            <= prr_limit
        )
    ].copy()

    fig, ax = plt.subplots(
        figsize=(10, 9)
    )

    ax.scatter(
        plot_data["ror"],
        plot_data["prr"],
        alpha=0.40,
        s=28
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(MIN_ROR, linestyle="--", linewidth=1.2)
    ax.axhline(MIN_PRR, linestyle="--", linewidth=1.2)

    ax.set_title(
        "ROR vs PRR Across FAERS Drug–Event Associations",
        pad=18,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Reporting Odds Ratio (ROR) — log scale"
    )

    ax.set_ylabel(
        "Proportional Reporting Ratio (PRR) — log scale"
    )

    ax.grid(
        alpha=0.20
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    save_figure(
        fig,
        "05_ror_vs_prr_log_scale"
    )


# ============================================================
# SAVE PAIR DATA
# ============================================================

def save_pair_dataset(
    pairs
):

    output = (
        TABLE_DIR
        / "drug_event_pairs.csv"
    )

    pairs.to_csv(
        output,
        index=False
    )

    print(
        f"\nSaved pair dataset: "
        f"{output}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 78)
    print("PHARMAAI - PRELIMINARY FAERS SIGNAL ANALYSIS")
    print("=" * 78)

    print(
        f"\nDashboard input:\n"
        f"{DASHBOARD_DIR}"
    )

    print(
        f"\nResults output:\n"
        f"{RESULTS_DIR}"
    )

    print(
        f"\nPoster graphs:\n"
        f"{POSTER_GRAPH_DIR}"
    )

    print(
        f"\nMinimum pair-report threshold: "
        f"{MIN_PAIR_REPORTS}"
    )

    dashboard = load_dashboard_data()

    pairs = build_drug_event_pairs(
        dashboard
    )

    save_pair_dataset(
        pairs
    )

    create_dataset_summary(
        dashboard,
        pairs
    )

    (
        drug_counts,
        event_counts,
        pair_counts
    ) = create_frequency_tables(
        pairs
    )

    (
        signals,
        filtered_signals,
        candidate_signals
    ) = calculate_signals(
        pairs,
        pair_counts
    )

    print("\n" + "=" * 78)
    print("CREATING POSTER-QUALITY FIGURES")
    print("=" * 78)

    plot_top_suspect_drugs(
        drug_counts
    )

    plot_top_adverse_events(
        event_counts
    )

    plot_top_drug_event_pairs(
        pair_counts
    )

    plot_top_ror_signals(
        candidate_signals
    )

    plot_ror_vs_prr(
        filtered_signals
    )

    print("\n" + "=" * 78)
    print("PRELIMINARY ANALYSIS COMPLETE")
    print("=" * 78)

    print(
        f"\nTables:\n"
        f"{TABLE_DIR}"
    )

    print(
        f"\nPoster graphs:\n"
        f"{POSTER_GRAPH_DIR}"
    )

    print(
        "\nIMPORTANT INTERPRETATION:"
        "\nROR and PRR indicate disproportional reporting."
        "\nThey do not establish that a drug caused an adverse event."
        "\nThese results should be described as preliminary candidate"
        "\nsafety signals / reporting associations."
    )


if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print("\n" + "=" * 78)
        print("ANALYSIS FAILED")
        print("=" * 78)

        print(
            f"\n{type(error).__name__}: "
            f"{error}"
        )

        sys.exit(1)
