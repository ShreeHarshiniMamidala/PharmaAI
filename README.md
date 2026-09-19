# PharmaSignal AI

## Machine Learning vs. Traditional Pharmacovigilance for Drug–Adverse-Event Signal Detection

**PharmaSignal AI** is a pharmacovigilance research project designed to identify and prioritize potential drug–adverse-event safety signals from the FDA Adverse Event Reporting System (FAERS).

The project establishes a traditional pharmacovigilance baseline using **Reporting Odds Ratio (ROR), Proportional Reporting Ratio (PRR), and Information Component (IC)-style measures**, and investigates whether machine-learning approaches can improve the identification and prioritization of potential safety signals.

> **Important:** Associations identified by this project represent reporting patterns in FAERS. They do not establish that a drug caused an adverse event and should not be interpreted as confirmed clinical safety signals.

---

## Research Problem

FAERS contains a large volume of real-world adverse-event reports, making it an important resource for post-market drug safety surveillance. However, missing information, duplicate records, inconsistent drug naming, and the large number of possible drug–event combinations make systematic signal detection and prioritization challenging.

Traditional disproportionality methods can identify unusual reporting patterns, but additional methods are needed to investigate whether potential signals can be prioritized more effectively.

PharmaSignal AI develops a reproducible pipeline for processing FAERS data, detecting reporting associations, prioritizing candidates for further review, and ultimately comparing traditional pharmacovigilance approaches with machine-learning models.

---

## Research Questions

### RQ1

**To what extent do machine-learning models improve the identification and prioritization of potential drug–adverse-event safety signals in FAERS compared with traditional disproportionality-based pharmacovigilance methods?**

### RQ2

**How do logistic regression, random forest, and gradient-boosting models compare with ROR, PRR, and IC in terms of precision, recall, F1-score, AUROC, PR-AUC, and Top-K ranking performance for potential drug–adverse-event safety signals?**

---

## Dataset

The preliminary study uses **2025 Q1–Q4 FAERS safety reports**.

The processed dataset contains:

| Statistic | Value |
|---|---:|
| Safety reports loaded | 1,396,239 |
| Reports with usable drug–event information | 1,390,248 |
| Report-level drug–event observations | 16,296,363 |
| Unique suspect drugs | 11,188 |
| Unique adverse-event terms | 16,541 |
| Unique drug–event associations | 1,476,886 |

FAERS reports may contain multiple suspect drugs and multiple adverse-event terms. Therefore, extracted drug–event combinations represent **report-level co-occurrences rather than confirmed causal relationships**.

---

## Data Preprocessing

The preprocessing pipeline extracts and standardizes information required for pharmacovigilance analysis.

Major steps include:

1. Load FAERS safety reports.
2. Extract suspect drugs and reported adverse events.
3. Remove reports without usable suspect-drug or adverse-event information.
4. Normalize relevant demographic and report-level fields.
5. Standardize drug and adverse-event representations where applicable.
6. Remove duplicate within-report drug–event contributions.
7. Construct report-level drug–event co-occurrences.
8. Aggregate drug–event reporting counts.
9. Apply minimum reporting-support criteria for preliminary analysis.
10. Generate analysis-ready tables for signal detection and visualization.

For the preliminary supported analysis, drug–event associations with at least **10 reports** are considered.

---

## Methodology

The overall PharmaSignal AI workflow is:

```text
FAERS Reports
      |
      v
Data Extraction
      |
      v
Cleaning & Standardization
      |
      v
Drug–Event Pair Construction
      |
      v
Reference Safety Labels
      |
      +-----------------------+
      |                       |
      v                       v
Traditional Methods       Machine Learning
ROR                       Logistic Regression
PRR                       Random Forest
IC                        Gradient Boosting
      |                       |
      +-----------+-----------+
                  |
                  v
          Common Evaluation
                  |
                  v
 Precision / Recall / F1
 AUROC / PR-AUC / Top-K
                  |
                  v
      Signal Prioritization
                  |
                  v
       PharmaSignal AI
          Dashboard
```

---

## Traditional Signal Detection

The preliminary analysis evaluates drug–event reporting associations using traditional disproportionality measures.

### Reporting Odds Ratio (ROR)

ROR compares the odds of reporting a particular adverse event for a drug with the odds of reporting that event for other drugs.

### Proportional Reporting Ratio (PRR)

PRR compares the proportion of a specific adverse event among reports involving a drug with the corresponding proportion among other reports.

### Information Component (IC)

An information-component-style observed-versus-expected measure is included in the current preliminary analysis.

> The current preliminary IC implementation should not be interpreted as a full Bayesian BCPNN/IC025 implementation.

---

## Preliminary Signal Screening

The preliminary analysis applies reporting-support and disproportionality criteria to reduce the influence of unstable sparse associations.

Current screening criteria include:

```text
Pair report count >= 10
ROR >= 2
ROR lower 95% CI > 1
PRR >= 2
IC >= 1
```

These thresholds are used as **preliminary research screening criteria** and should not be interpreted as universal clinical or regulatory definitions of a safety signal.

---

## Signal Prioritization

PharmaSignal AI extends signal screening with a preliminary prioritization framework.

Drug–event associations are organized using:

- Reporting support
- Disproportionality strength
- ROR confidence intervals
- PRR
- IC-style evidence

The priority matrix categorizes associations into groups such as:

- **Priority Review**
- **Strong Signal / Lower Support**
- **High Support / Moderate Signal**
- **Monitor**

Priority represents **relative priority for further pharmacovigilance investigation**, not clinical risk or evidence that a medication caused an adverse event.

---

## Preliminary Results

Analysis of approximately **1.39 million FAERS reports** produced approximately **1.48 million unique drug–event reporting associations**.

Preliminary findings show that:

- Large-scale FAERS drug–event associations can be systematically extracted and aggregated.
- Reporting-support criteria help reduce the influence of sparse associations.
- ROR and PRR demonstrate substantial agreement across many supported associations.
- Disproportionality strength can be combined with reporting support to prioritize associations for further investigation.
- The resulting traditional analysis establishes a baseline for subsequent machine-learning comparison.

---

## Visualizations

The preliminary analysis generates poster-ready visualizations including:

### Most Frequently Reported Drug–Event Pairs

Displays high-frequency suspect drug–adverse-event reporting associations in the 2025 FAERS dataset.

### Drug–Event Signal Prioritization Map

Visualizes supported associations according to reporting support and disproportionality strength to identify candidates for further review.

### ROR vs. PRR Density

Examines the relationship between two traditional disproportionality measures across supported drug–event associations.

---

## Machine-Learning Models

The planned machine-learning comparison includes:

- **Logistic Regression**
- **Random Forest**
- **Gradient Boosting**

The ML models will be evaluated against the traditional pharmacovigilance baseline using the same reference-labeled evaluation framework.

### Evaluation Metrics

The planned evaluation includes:

- Precision
- Recall
- F1-score
- AUROC
- PR-AUC
- Top-K ranking performance

The machine-learning comparison is currently **future work** and no ML performance results are reported as completed results at this stage.

---

## Project Status

### Completed / Preliminary

- FAERS 2025 data extraction
- Data cleaning and preprocessing
- Suspect drug–event pair construction
- Reporting-frequency analysis
- ROR calculation
- ROR 95% confidence intervals
- PRR calculation
- Preliminary IC-style calculation
- Supported signal screening
- Preliminary signal prioritization
- Poster-ready visualizations

### In Progress / Future Work

- Reference safety-association label construction
- Final IC methodology
- Logistic regression
- Random forest
- Gradient boosting
- ML vs. traditional-method evaluation
- Precision, recall, F1, AUROC and PR-AUC comparison
- Top-K ranking evaluation
- Final interactive dashboard

---

## Repository Structure

```text
PharmaAI/
|
├── data/
│   ├── raw/
│   └── processed/
|
├── results/
│   └── preliminary/
│       ├── tables/
│       └── poster_graphs/
|
├── scripts/
│   ├── data_processing/
│   ├── signal_analysis/
│   └── visualization/
|
├── README.md
└── requirements.txt
```

The exact structure may evolve as the machine-learning and dashboard components are added.

---

## Future Work

Future work will incorporate appropriate reference safety-association labels and develop **logistic regression, random forest, and gradient-boosting models** for signal identification and prioritization.

The ML approaches will be compared with **ROR, PRR, and IC** using precision, recall, F1-score, AUROC, PR-AUC, and Top-K ranking metrics. The final framework will integrate signal detection, prioritization, model comparison, and interpretable visualization into the PharmaSignal AI dashboard.

---

## Limitations

FAERS is a spontaneous reporting system and is subject to limitations including under-reporting, duplicate reports, missing information, reporting bias, inconsistent drug terminology, and the absence of reliable exposure denominators.

Disproportionate reporting therefore does not establish incidence, prevalence, clinical risk, or causality. Results from PharmaSignal AI should be interpreted as **hypothesis-generating reporting patterns requiring further expert investigation**.

---

## Disclaimer

**PharmaSignal AI is a research project and is not a clinical decision-support or diagnostic system.**

A detected drug–event association does not demonstrate that the drug caused the reported event. All identified patterns require appropriate clinical, epidemiological, and pharmacovigilance review before conclusions about drug safety can be made.
