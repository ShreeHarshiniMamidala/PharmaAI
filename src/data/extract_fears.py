from pathlib import Path
import json
import re
import zipfile
import traceback
import gc

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================
# PHARMAAI - FAERS CANONICAL EXTRACTION
# ============================================================
#
# RESEARCH DESIGN
# ---------------
# 2025 Q1-Q4 -> Development data
# 2026 Q1    -> Temporal test data
#
# Reserved:
# 2024       -> Historical extension
# 2026 Q2    -> Future robustness experiment
#
# UNIT OF OBSERVATION
# -------------------
# One row =
# safetyreportid × suspect_drug × adverse_event
#
# Only drugs with:
# drugcharacterization == 1
# are treated as suspect drugs.
#
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\mamid\OneDrive\Desktop\PharmaAI"
)

RAW_FAERS_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "faers"
)

INTERIM_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "extraction_chunks"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

INTERIM_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. QUARTERS
# ============================================================

DEVELOPMENT_QUARTERS = [
    "2025_Q1",
    "2025_Q2",
    "2025_Q3",
    "2025_Q4",
]

TEST_QUARTERS = [
    "2026_Q1",
]

PROCESS_QUARTERS = (
    DEVELOPMENT_QUARTERS
    + TEST_QUARTERS
)


# ============================================================
# 3. FINAL CANONICAL OUTPUTS
# ============================================================

DEVELOPMENT_OUTPUT = (
    PROCESSED_DIR
    / "faers_2025_clean.parquet"
)

TEST_OUTPUT = (
    PROCESSED_DIR
    / "faers_2026_Q1_clean.parquet"
)


# ============================================================
# 4. OPTIONS
# ============================================================

# Full processing now
TEST_MODE = False

# Resume:
# existing successful chunks will be skipped
SKIP_EXISTING_CHUNKS = True

# Keep chunks until we validate final outputs
DELETE_CHUNKS_AFTER_MERGE = False


# ============================================================
# 5. BASIC HELPERS
# ============================================================

def safe_get(dictionary, key, default=None):

    if not isinstance(dictionary, dict):
        return default

    return dictionary.get(
        key,
        default
    )


def safe_float(value):

    if value is None:
        return None

    try:
        return float(value)

    except (TypeError, ValueError):
        return None


def normalize_age(age, unit):

    age = safe_float(age)

    if age is None:
        return None

    if unit is None:
        return None

    unit = str(unit)

    # Decade
    if unit == "800":
        return age * 10

    # Year
    if unit == "801":
        return age

    # Month
    if unit == "802":
        return age / 12

    # Week
    if unit == "803":
        return age / 52.1429

    # Day
    if unit == "804":
        return age / 365.25

    # Hour
    if unit == "805":
        return age / (
            365.25 * 24
        )

    return None


# ============================================================
# 6. TEXT NORMALIZATION
# ============================================================

def normalize_text(value):

    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value


def normalize_drug_name(value):

    value = normalize_text(value)

    if value is None:
        return None

    return value.upper()


def normalize_event_name(value):

    value = normalize_text(value)

    if value is None:
        return None

    return value.upper()


# ============================================================
# 7. DRUG METADATA HELPERS
# ============================================================

def get_active_substance(drug):

    active = safe_get(
        drug,
        "activesubstance",
        {}
    )

    if not isinstance(active, dict):
        return None

    return normalize_text(
        safe_get(
            active,
            "activesubstancename"
        )
    )


def get_openfda_generic_name(drug):

    openfda = safe_get(
        drug,
        "openfda",
        {}
    )

    if not isinstance(openfda, dict):
        return None

    names = safe_get(
        openfda,
        "generic_name",
        []
    )

    if not names:
        return None

    if not isinstance(names, list):
        names = [names]

    cleaned = []

    for name in names:

        name = normalize_text(name)

        if name:
            cleaned.append(name)

    if not cleaned:
        return None

    return " | ".join(
        sorted(set(cleaned))
    )


# ============================================================
# 8. EXTRACT ONE REPORT
# ============================================================

def extract_report_observations(
    report,
    year,
    quarter
):

    patient = safe_get(
        report,
        "patient",
        {}
    )

    if not isinstance(patient, dict):
        return []


    # --------------------------------------------------------
    # Report ID
    # --------------------------------------------------------

    safetyreportid = normalize_text(
        safe_get(
            report,
            "safetyreportid"
        )
    )

    if safetyreportid is None:
        return []


    # --------------------------------------------------------
    # Report information
    # --------------------------------------------------------

    serious = normalize_text(
        safe_get(
            report,
            "serious"
        )
    )

    country = (
        safe_get(
            report,
            "occurcountry"
        )
        or
        safe_get(
            report,
            "primarysourcecountry"
        )
    )

    country = normalize_text(
        country
    )

    report_type = normalize_text(
        safe_get(
            report,
            "reporttype"
        )
    )

    receive_date = normalize_text(
        safe_get(
            report,
            "receivedate"
        )
    )

    receipt_date = normalize_text(
        safe_get(
            report,
            "receiptdate"
        )
    )


    # --------------------------------------------------------
    # Patient
    # --------------------------------------------------------

    patient_age = normalize_age(

        safe_get(
            patient,
            "patientonsetage"
        ),

        safe_get(
            patient,
            "patientonsetageunit"
        )
    )

    patient_sex = normalize_text(
        safe_get(
            patient,
            "patientsex"
        )
    )


    # --------------------------------------------------------
    # Reporter
    # --------------------------------------------------------

    primary_source = safe_get(
        report,
        "primarysource",
        {}
    )

    reporter_qualification = None

    if isinstance(
        primary_source,
        dict
    ):

        reporter_qualification = (
            normalize_text(
                safe_get(
                    primary_source,
                    "qualification"
                )
            )
        )


    # ========================================================
    # ADVERSE EVENTS
    # ========================================================

    reactions = safe_get(
        patient,
        "reaction",
        []
    )

    if not isinstance(reactions, list):
        reactions = []

    adverse_events = set()

    for reaction in reactions:

        if not isinstance(
            reaction,
            dict
        ):
            continue

        event = normalize_event_name(
            safe_get(
                reaction,
                "reactionmeddrapt"
            )
        )

        if event:
            adverse_events.add(
                event
            )


    # No usable event
    if not adverse_events:
        return []


    # ========================================================
    # DRUGS
    # ========================================================

    drugs = safe_get(
        patient,
        "drug",
        []
    )

    if not isinstance(drugs, list):
        drugs = []

    num_drugs = len(drugs)

    # Unique reaction count
    num_reactions = len(
        adverse_events
    )

    suspect_drugs = []


    for drug in drugs:

        if not isinstance(
            drug,
            dict
        ):
            continue


        # ----------------------------------------------------
        # 1 = suspect
        # 2 = concomitant
        # 3 = interacting
        # ----------------------------------------------------

        characterization = safe_get(
            drug,
            "drugcharacterization"
        )

        if characterization is None:
            continue

        if str(characterization) != "1":
            continue


        # ----------------------------------------------------
        # Medicinal product
        # ----------------------------------------------------

        medicinal_product = (
            normalize_drug_name(
                safe_get(
                    drug,
                    "medicinalproduct"
                )
            )
        )

        if medicinal_product is None:
            continue


        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        active_substance = (
            get_active_substance(
                drug
            )
        )

        generic_name = (
            get_openfda_generic_name(
                drug
            )
        )

        drug_indication = (
            normalize_text(
                safe_get(
                    drug,
                    "drugindication"
                )
            )
        )

        drug_route = (
            normalize_text(
                safe_get(
                    drug,
                    "drugadministrationroute"
                )
            )
        )

        dosage_form = (
            normalize_text(
                safe_get(
                    drug,
                    "drugdosageform"
                )
            )
        )


        suspect_drugs.append(
            {
                "suspect_drug":
                    medicinal_product,

                "active_substance":
                    active_substance,

                "generic_name":
                    generic_name,

                "drug_indication":
                    drug_indication,

                "drug_route":
                    drug_route,

                "drug_dosage_form":
                    dosage_form,
            }
        )


    # No usable suspect drug
    if not suspect_drugs:
        return []


    # ========================================================
    # DRUG × EVENT EXPANSION
    # ========================================================

    observations = []

    seen_pairs = set()


    for drug in suspect_drugs:

        suspect_drug = (
            drug[
                "suspect_drug"
            ]
        )


        for adverse_event in adverse_events:

            pair_key = (
                safetyreportid,
                suspect_drug,
                adverse_event
            )


            # Prevent duplicate report/drug/event
            if pair_key in seen_pairs:
                continue


            seen_pairs.add(
                pair_key
            )


            observations.append(
                {

                    # -----------------------------------------
                    # Report ID
                    # -----------------------------------------

                    "safetyreportid":
                        safetyreportid,


                    # -----------------------------------------
                    # Time
                    # -----------------------------------------

                    "year":
                        year,

                    "quarter":
                        quarter,

                    "receivedate":
                        receive_date,

                    "receiptdate":
                        receipt_date,


                    # -----------------------------------------
                    # Core signal variables
                    # -----------------------------------------

                    "suspect_drug":
                        suspect_drug,

                    "adverse_event":
                        adverse_event,


                    # -----------------------------------------
                    # Drug metadata
                    # -----------------------------------------

                    "active_substance":
                        drug[
                            "active_substance"
                        ],

                    "generic_name":
                        drug[
                            "generic_name"
                        ],

                    "drug_indication":
                        drug[
                            "drug_indication"
                        ],

                    "drug_route":
                        drug[
                            "drug_route"
                        ],

                    "drug_dosage_form":
                        drug[
                            "drug_dosage_form"
                        ],


                    # -----------------------------------------
                    # Report metadata
                    # -----------------------------------------

                    "serious":
                        serious,

                    "patient_age_years":
                        patient_age,

                    "patient_sex":
                        patient_sex,

                    "country":
                        country,

                    "report_type":
                        report_type,

                    "reporter_qualification":
                        reporter_qualification,


                    # -----------------------------------------
                    # Complexity
                    # -----------------------------------------

                    "num_drugs":
                        num_drugs,

                    "num_reactions":
                        num_reactions,
                }
            )


    return observations


# ============================================================
# 9. CHUNK PATH
# ============================================================

def get_chunk_path(
    quarter_name,
    zip_path
):

    quarter_dir = (
        INTERIM_DIR
        / quarter_name
    )

    quarter_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    output_name = (
        zip_path.name
        .replace(
            ".json.zip",
            ""
        )
        .replace(
            ".zip",
            ""
        )
    )


    return (
        quarter_dir
        / f"{output_name}.parquet"
    )


# ============================================================
# 10. PROCESS ONE ZIP
# ============================================================

def process_zip(
    zip_path,
    quarter_name
):

    chunk_path = get_chunk_path(
        quarter_name,
        zip_path
    )


    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    if (
        SKIP_EXISTING_CHUNKS
        and
        chunk_path.exists()
    ):

        print(
            f"[SKIP] "
            f"{quarter_name} / "
            f"{zip_path.name}"
        )

        return {
            "status": "skipped",
            "reports": 0,
            "usable_reports": 0,
            "observations": 0,
            "errors": 0
        }


    print("\n" + "=" * 75)

    print(
        f"Processing: "
        f"{quarter_name} / "
        f"{zip_path.name}"
    )

    print("=" * 75)


    year = int(
        quarter_name.split("_")[0]
    )

    quarter = (
        quarter_name
        .split("_")[1]
    )


    observations = []

    reports_seen = 0

    usable_reports = 0

    report_errors = 0


    try:

        # ====================================================
        # OPEN ZIP
        # ====================================================

        with zipfile.ZipFile(
            zip_path,
            "r"
        ) as archive:


            json_files = [

                name

                for name
                in archive.namelist()

                if name
                .lower()
                .endswith(".json")
            ]


            if not json_files:

                print(
                    "❌ No JSON found "
                    "inside ZIP."
                )

                return {
                    "status": "failed",
                    "reports": 0,
                    "usable_reports": 0,
                    "observations": 0,
                    "errors": 1
                }


            # =================================================
            # JSON FILE LOOP
            # =================================================

            for json_name in json_files:

                print(
                    f"Reading: "
                    f"{json_name}"
                )


                with archive.open(
                    json_name
                ) as file:

                    data = json.load(
                        file
                    )


                reports = data.get(
                    "results",
                    []
                )


                print(
                    f"Reports found: "
                    f"{len(reports):,}"
                )


                # =============================================
                # REPORT LOOP
                # =============================================

                for report in reports:

                    reports_seen += 1


                    try:

                        rows = (
                            extract_report_observations(
                                report,
                                year,
                                quarter
                            )
                        )


                        if rows:

                            usable_reports += 1

                            observations.extend(
                                rows
                            )


                    except Exception:

                        report_errors += 1


        # ====================================================
        # DATAFRAME
        # ====================================================

        if not observations:

            print(
                "⚠️ No usable observations."
            )

            return {
                "status": "failed",
                "reports": reports_seen,
                "usable_reports": 0,
                "observations": 0,
                "errors": report_errors
            }


        df = pd.DataFrame(
            observations
        )


        # Release list memory
        del observations

        gc.collect()


        # ====================================================
        # CHUNK-LEVEL DEDUPLICATION
        # ====================================================

        before = len(df)


        df = df.drop_duplicates(
            subset=[
                "safetyreportid",
                "suspect_drug",
                "adverse_event"
            ],
            keep="first"
        )


        after = len(df)

        removed = (
            before - after
        )


        # ====================================================
        # SAVE CHUNK
        # ====================================================

        df.to_parquet(
            chunk_path,
            index=False,
            engine="pyarrow",
            compression="snappy"
        )


        print(
            "\n✅ Completed"
        )

        print(
            f"Reports seen        : "
            f"{reports_seen:,}"
        )

        print(
            f"Usable reports      : "
            f"{usable_reports:,}"
        )

        print(
            f"Drug-event rows     : "
            f"{after:,}"
        )

        print(
            f"Duplicates removed  : "
            f"{removed:,}"
        )

        print(
            f"Report errors       : "
            f"{report_errors:,}"
        )

        print(
            f"Chunk:\n"
            f"{chunk_path}"
        )


        del df

        gc.collect()


        return {
            "status": "success",
            "reports": reports_seen,
            "usable_reports": usable_reports,
            "observations": after,
            "errors": report_errors
        }


    except Exception as error:

        print(
            f"\n❌ ERROR processing "
            f"{zip_path.name}"
        )

        print(
            error
        )

        traceback.print_exc()


        return {
            "status": "failed",
            "reports": reports_seen,
            "usable_reports": usable_reports,
            "observations": 0,
            "errors": report_errors + 1
        }


# ============================================================
# 11. PROCESS SELECTED QUARTERS
# ============================================================

def process_all_selected_quarters():

    print("\n" + "=" * 75)

    print(
        "PHARMAAI FULL FAERS EXTRACTION"
    )

    print("=" * 75)


    print(
        "\nDEVELOPMENT DATA:"
    )

    for quarter in DEVELOPMENT_QUARTERS:

        print(
            f"  {quarter}"
        )


    print(
        "\nTEMPORAL TEST DATA:"
    )

    for quarter in TEST_QUARTERS:

        print(
            f"  {quarter}"
        )


    print(
        "\nNOT PROCESSED:"
    )

    print(
        "  2024_Q1-Q4"
    )

    print(
        "  2026_Q2"
    )


    total_files = 0

    successful_files = 0

    skipped_files = 0

    failed_files = 0

    total_reports = 0

    total_usable_reports = 0

    total_observations = 0

    total_errors = 0


    # ========================================================
    # QUARTER LOOP
    # ========================================================

    for quarter_name in PROCESS_QUARTERS:


        quarter_dir = (
            RAW_FAERS_DIR
            / quarter_name
        )


        if not quarter_dir.exists():

            print(
                f"\n❌ Quarter folder "
                f"not found:"
            )

            print(
                quarter_dir
            )

            failed_files += 1

            continue


        zip_files = sorted(
            quarter_dir.glob(
                "*.zip"
            )
        )


        print("\n" + "#" * 75)

        print(
            quarter_name
        )

        print(
            f"ZIP files found: "
            f"{len(zip_files):,}"
        )

        print("#" * 75)


        total_files += (
            len(zip_files)
        )


        # ====================================================
        # ZIP LOOP
        # ====================================================

        for index, zip_path in enumerate(
            zip_files,
            start=1
        ):


            print(
                f"\n[{index}/"
                f"{len(zip_files)}]"
            )


            result = process_zip(
                zip_path,
                quarter_name
            )


            status = (
                result[
                    "status"
                ]
            )


            if status == "success":

                successful_files += 1


            elif status == "skipped":

                skipped_files += 1


            else:

                failed_files += 1


            total_reports += (
                result[
                    "reports"
                ]
            )


            total_usable_reports += (
                result[
                    "usable_reports"
                ]
            )


            total_observations += (
                result[
                    "observations"
                ]
            )


            total_errors += (
                result[
                    "errors"
                ]
            )


    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 75)

    print(
        "EXTRACTION SUMMARY"
    )

    print("=" * 75)


    print(
        f"ZIP files selected : "
        f"{total_files:,}"
    )

    print(
        f"Successful         : "
        f"{successful_files:,}"
    )

    print(
        f"Already processed  : "
        f"{skipped_files:,}"
    )

    print(
        f"Failed             : "
        f"{failed_files:,}"
    )

    print(
        f"Reports processed  : "
        f"{total_reports:,}"
    )

    print(
        f"Usable reports     : "
        f"{total_usable_reports:,}"
    )

    print(
        f"Drug-event rows    : "
        f"{total_observations:,}"
    )

    print(
        f"Report errors      : "
        f"{total_errors:,}"
    )


    if failed_files > 0:

        print(
            "\n⚠️ Extraction had failures."
        )

        print(
            "Run this script again "
            "after checking the errors."
        )

        print(
            "Existing successful chunks "
            "will be skipped."
        )

        return False


    print(
        "\n✅ All selected FAERS ZIP "
        "files have extraction chunks."
    )


    return True


# ============================================================
# 12. COMBINE CHUNKS SAFELY
# ============================================================

def combine_chunks_streaming(
    quarters,
    output_file,
    dataset_name
):
    """
    Combine extraction chunks into ONE canonical Parquet file.

    Uses PyArrow ParquetWriter instead of pd.concat() so we do
    not load all ~millions of observations into RAM at once.

    NOTE:
    Within-report duplicates have already been prevented during
    extraction. Global duplicate checking will be performed in
    the validation / pair-building stage.
    """

    print("\n" + "=" * 75)

    print(
        f"BUILDING CANONICAL DATASET:"
        f"\n{dataset_name}"
    )

    print("=" * 75)


    chunk_files = []


    for quarter in quarters:

        quarter_dir = (
            INTERIM_DIR
            / quarter
        )


        if not quarter_dir.exists():

            print(
                f"⚠️ No chunk directory:"
                f"\n{quarter_dir}"
            )

            continue


        files = sorted(
            quarter_dir.glob(
                "*.parquet"
            )
        )


        chunk_files.extend(
            files
        )


    if not chunk_files:

        print(
            "❌ No extraction chunks found."
        )

        return False


    print(
        f"Chunks found: "
        f"{len(chunk_files):,}"
    )


    # --------------------------------------------------------
    # Remove old canonical output before rebuilding
    # --------------------------------------------------------

    if output_file.exists():

        print(
            f"\nRemoving previous canonical file:"
            f"\n{output_file}"
        )

        output_file.unlink()


    writer = None

    total_rows = 0


    try:

        for index, chunk_file in enumerate(
            chunk_files,
            start=1
        ):


            print(
                f"Writing chunk "
                f"{index}/{len(chunk_files)}: "
                f"{chunk_file.name}"
            )


            table = pq.read_table(
                chunk_file
            )


            if writer is None:

                writer = pq.ParquetWriter(
                    output_file,
                    table.schema,
                    compression="snappy"
                )


            writer.write_table(
                table
            )


            total_rows += (
                table.num_rows
            )


            del table

            gc.collect()


        if writer is not None:

            writer.close()

            writer = None


        print(
            f"\n✅ Canonical dataset created:"
        )

        print(
            output_file
        )

        print(
            f"\nRows written: "
            f"{total_rows:,}"
        )


        return True


    except Exception as error:

        if writer is not None:

            writer.close()


        print(
            f"\n❌ Error combining chunks:"
        )

        print(
            error
        )

        traceback.print_exc()


        return False


# ============================================================
# 13. FINAL DATASET SUMMARY
# ============================================================

def summarize_canonical_file(
    file_path,
    dataset_name
):

    print("\n" + "=" * 75)

    print(
        f"CANONICAL DATASET SUMMARY:"
        f"\n{dataset_name}"
    )

    print("=" * 75)


    if not file_path.exists():

        print(
            "❌ File not found:"
        )

        print(
            file_path
        )

        return


    parquet_file = pq.ParquetFile(
        file_path
    )


    total_rows = (
        parquet_file.metadata.num_rows
    )


    size_gb = (
        file_path.stat().st_size
        / (1024 ** 3)
    )


    print(
        f"Rows      : "
        f"{total_rows:,}"
    )

    print(
        f"File size : "
        f"{size_gb:.2f} GB"
    )

    print(
        f"Path:"
        f"\n{file_path}"
    )


# ============================================================
# 14. FULL PIPELINE
# ============================================================

def run_full():


    extraction_success = (
        process_all_selected_quarters()
    )


    if not extraction_success:

        print(
            "\n❌ Final canonical datasets "
            "were not rebuilt because "
            "extraction failures occurred."
        )

        return


    # ========================================================
    # 2025 DEVELOPMENT DATA
    # ========================================================

    development_success = (
        combine_chunks_streaming(

            quarters=
                DEVELOPMENT_QUARTERS,

            output_file=
                DEVELOPMENT_OUTPUT,

            dataset_name=
                "2025 DEVELOPMENT DATA"
        )
    )


    # ========================================================
    # 2026 Q1 TEST DATA
    # ========================================================

    test_success = (
        combine_chunks_streaming(

            quarters=
                TEST_QUARTERS,

            output_file=
                TEST_OUTPUT,

            dataset_name=
                "2026 Q1 TEMPORAL TEST DATA"
        )
    )


    # ========================================================
    # SUMMARIES
    # ========================================================

    if development_success:

        summarize_canonical_file(

            DEVELOPMENT_OUTPUT,

            "2025 DEVELOPMENT DATA"
        )


    if test_success:

        summarize_canonical_file(

            TEST_OUTPUT,

            "2026 Q1 TEMPORAL TEST DATA"
        )


    # ========================================================
    # CHUNK CLEANUP
    # ========================================================

    if (
        DELETE_CHUNKS_AFTER_MERGE
        and
        development_success
        and
        test_success
    ):

        import shutil

        print(
            "\nDeleting temporary "
            "extraction chunks..."
        )


        shutil.rmtree(
            INTERIM_DIR
        )


        print(
            "Temporary chunks deleted."
        )


# ============================================================
# 15. MAIN
# ============================================================

if __name__ == "__main__":


    print("=" * 75)

    print(
        "PharmaAI - Full FAERS "
        "Drug-Event Extraction"
    )

    print("=" * 75)


    print(
        "\nProject root:"
    )

    print(
        PROJECT_ROOT
    )


    print(
        "\nRaw FAERS:"
    )

    print(
        RAW_FAERS_DIR
    )


    print(
        "\nTemporary chunks:"
    )

    print(
        INTERIM_DIR
    )


    print(
        "\nFinal datasets:"
    )

    print(
        DEVELOPMENT_OUTPUT
    )

    print(
        TEST_OUTPUT
    )


    # --------------------------------------------------------
    # Verify raw directory
    # --------------------------------------------------------

    if not RAW_FAERS_DIR.exists():

        print(
            "\n❌ Raw FAERS directory "
            "was not found."
        )

        raise SystemExit


    # --------------------------------------------------------
    # Run full extraction
    # --------------------------------------------------------

    run_full()


    print("\n" + "=" * 75)

    print(
        "PharmaAI extraction finished."
    )

    print("=" * 75)