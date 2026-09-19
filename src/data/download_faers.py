"""
PharmaAI - FDA FAERS/openFDA Drug Adverse Event Downloader

Downloads Human Drug Adverse Event data from the official FDA
download index for the years 2024, 2025, and 2026.

Source:
https://api.fda.gov/download.json

The script:
1. Downloads the official FDA download index
2. Finds the Human Drug / Adverse Event endpoint
3. Selects partitions from 2024, 2025, and 2026
4. Creates one folder per quarter
5. Downloads the ZIP files
6. Supports resume/restart
7. Skips files that already exist
8. Saves a download manifest
9. Does NOT download unrelated FDA datasets
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, List

import requests
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

FDA_INDEX_URL = "https://api.fda.gov/download.json"

# Years we want
TARGET_YEARS = {2024, 2025, 2026}

# Project root:
# PharmaAI/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Raw FAERS data directory
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "faers"

# Manifest showing exactly what was downloaded
MANIFEST_FILE = OUTPUT_DIR / "download_manifest.json"

# Temporary files while downloading
TEMP_SUFFIX = ".part"

# Network configuration
CHUNK_SIZE = 1024 * 1024  # 1 MB
TIMEOUT = 60
MAX_RETRIES = 5

HEADERS = {
    "User-Agent": "PharmaAI-Research-Downloader/1.0"
}


# ============================================================
# HELPERS
# ============================================================

def get_fda_download_index() -> dict:
    """
    Download the official FDA download index.
    """

    print("\n" + "=" * 70)
    print("Downloading FDA download index")
    print("=" * 70)

    response = requests.get(
        FDA_INDEX_URL,
        headers=HEADERS,
        timeout=TIMEOUT
    )

    response.raise_for_status()

    data = response.json()

    print(f"FDA index retrieved successfully.")
    print(f"Index last updated: {data.get('meta', {}).get('last_updated')}")

    return data


def extract_drug_event_partitions(data: dict) -> List[dict]:
    """
    Extract Human Drug Adverse Event partitions.

    Expected structure:

    results
        -> drug
            -> event
                -> partitions
    """

    try:
        partitions = data["results"]["drug"]["event"]["partitions"]
    except KeyError as exc:
        raise RuntimeError(
            "Could not find results -> drug -> event -> partitions "
            "in FDA download.json"
        ) from exc

    print(f"\nTotal FDA Drug Event partitions in index: {len(partitions):,}")

    return partitions


def get_year_from_partition(partition: dict):
    """
    Extract year from display_name.

    Example:
        '2026 Q1 (part 1 of 31)'
        -> 2026
    """

    display_name = partition.get("display_name", "")

    match = re.match(r"(\d{4})\s+Q[1-4]", display_name)

    if not match:
        return None

    return int(match.group(1))


def filter_target_partitions(partitions: List[dict]) -> List[dict]:
    """
    Keep only 2024, 2025, and 2026 Drug Event partitions.
    """

    selected = []

    for partition in partitions:

        year = get_year_from_partition(partition)

        if year in TARGET_YEARS:
            selected.append(partition)

    return selected


def extract_quarter(display_name: str) -> str:
    """
    Extract quarter from display name.

    Example:
        2026 Q1 (part 1 of 31)
        -> 2026_Q1
    """

    match = re.match(r"(\d{4})\s+(Q[1-4])", display_name)

    if not match:
        return "unknown"

    return f"{match.group(1)}_{match.group(2)}"


def safe_filename(url: str) -> str:
    """
    Extract filename from URL.
    """

    return url.rstrip("/").split("/")[-1]


def create_output_directory(quarter: str) -> Path:
    """
    Create:

    data/raw/faers/2024_Q1/
    data/raw/faers/2024_Q2/
    ...
    """

    directory = OUTPUT_DIR / quarter
    directory.mkdir(parents=True, exist_ok=True)

    return directory


def download_file(
    url: str,
    destination: Path,
    max_retries: int = MAX_RETRIES
) -> bool:
    """
    Download a file with resume support.

    Returns:
        True  -> downloaded successfully
        False -> failed
    """

    temp_file = destination.with_suffix(
        destination.suffix + TEMP_SUFFIX
    )

    # --------------------------------------------------------
    # Already downloaded
    # --------------------------------------------------------

    if destination.exists() and destination.stat().st_size > 0:

        print(f"\n[SKIP] Already exists:")
        print(f"       {destination}")

        return True

    # --------------------------------------------------------
    # Determine resume position
    # --------------------------------------------------------

    resume_position = 0

    if temp_file.exists():
        resume_position = temp_file.stat().st_size

    for attempt in range(1, max_retries + 1):

        try:

            headers = HEADERS.copy()

            if resume_position > 0:
                headers["Range"] = f"bytes={resume_position}-"

            print(
                f"\n[DOWNLOAD] Attempt {attempt}/{max_retries}"
            )
            print(f"URL: {url}")
            print(f"Destination: {destination}")

            response = requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=TIMEOUT
            )

            # ------------------------------------------------
            # Server accepted resume request
            # ------------------------------------------------

            if resume_position > 0 and response.status_code == 206:

                mode = "ab"

                content_length = response.headers.get(
                    "Content-Length"
                )

                if content_length:
                    total_size = (
                        resume_position + int(content_length)
                    )
                else:
                    total_size = None

                initial_position = resume_position

            # ------------------------------------------------
            # Server ignored Range request
            # ------------------------------------------------

            elif response.status_code == 200:

                mode = "wb"

                content_length = response.headers.get(
                    "Content-Length"
                )

                total_size = (
                    int(content_length)
                    if content_length
                    else None
                )

                initial_position = 0
                resume_position = 0

            else:

                response.raise_for_status()

            # ------------------------------------------------
            # Download
            # ------------------------------------------------

            with open(temp_file, mode) as f:

                progress = tqdm(
                    total=total_size,
                    initial=initial_position,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=destination.name
                )

                for chunk in response.iter_content(
                    chunk_size=CHUNK_SIZE
                ):

                    if chunk:

                        f.write(chunk)
                        progress.update(len(chunk))

                progress.close()

            # ------------------------------------------------
            # Move completed file into place
            # ------------------------------------------------

            temp_file.replace(destination)

            print(f"[OK] {destination}")

            return True

        except Exception as exc:

            print(
                f"\n[ERROR] Download failed: {exc}"
            )

            if attempt < max_retries:

                wait_time = 5 * attempt

                print(
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(wait_time)

                # Recalculate resume position
                if temp_file.exists():
                    resume_position = temp_file.stat().st_size

            else:

                print(
                    f"[FAILED] Could not download:"
                    f"\n{url}"
                )

                return False

    return False


def calculate_summary(partitions: List[dict]) -> Dict:
    """
    Calculate number of files, records and approximate size.
    """

    total_files = len(partitions)

    total_records = sum(
        int(p.get("records", 0))
        for p in partitions
    )

    total_size_mb = sum(
        float(p.get("size_mb", 0))
        for p in partitions
    )

    quarters = {}

    for partition in partitions:

        quarter = extract_quarter(
            partition.get("display_name", "")
        )

        if quarter not in quarters:

            quarters[quarter] = {
                "files": 0,
                "records": 0,
                "size_mb": 0.0
            }

        quarters[quarter]["files"] += 1

        quarters[quarter]["records"] += int(
            partition.get("records", 0)
        )

        quarters[quarter]["size_mb"] += float(
            partition.get("size_mb", 0)
        )

    return {
        "total_files": total_files,
        "total_records": total_records,
        "total_size_mb": total_size_mb,
        "quarters": quarters
    }


def save_manifest(
    selected_partitions: List[dict],
    results: List[dict],
    index_last_updated: str
):
    """
    Save a record of what FDA files were selected and
    whether each file was successfully downloaded.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source": FDA_INDEX_URL,
        "index_last_updated": index_last_updated,
        "target_years": sorted(TARGET_YEARS),
        "endpoint": "/drug/event",
        "files": results
    }

    with open(
        MANIFEST_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2
        )

    print(
        f"\nManifest saved to:\n{MANIFEST_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("PHARMAAI - FDA FAERS DOWNLOADER")
    print("=" * 70)

    print("\nTarget years:")
    print(", ".join(str(y) for y in sorted(TARGET_YEARS)))

    print("\nEndpoint:")
    print("/drug/event")

    print("\nOutput:")
    print(OUTPUT_DIR)

    # --------------------------------------------------------
    # 1. Get FDA index
    # --------------------------------------------------------

    data = get_fda_download_index()

    # --------------------------------------------------------
    # 2. Get Drug Event partitions
    # --------------------------------------------------------

    all_partitions = extract_drug_event_partitions(data)

    # --------------------------------------------------------
    # 3. Filter 2024-2026
    # --------------------------------------------------------

    selected = filter_target_partitions(
        all_partitions
    )

    if not selected:

        raise RuntimeError(
            "No 2024-2026 Drug Event partitions found."
        )

    # --------------------------------------------------------
    # 4. Summary
    # --------------------------------------------------------

    summary = calculate_summary(selected)

    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)

    print(
        f"Files:       {summary['total_files']:,}"
    )

    print(
        f"Records:     {summary['total_records']:,}"
    )

    print(
        f"Approx size: {summary['total_size_mb']:,.2f} MB"
    )

    print(
        f"Approx size: "
        f"{summary['total_size_mb'] / 1024:,.2f} GB"
    )

    print("\nBy quarter:")

    for quarter in sorted(summary["quarters"]):

        q = summary["quarters"][quarter]

        print(
            f"  {quarter}: "
            f"{q['files']} files | "
            f"{q['records']:,} records | "
            f"{q['size_mb']:,.2f} MB"
        )

    # --------------------------------------------------------
    # 5. Confirm
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    answer = input(
        "Start downloading these files? [Y/n]: "
    ).strip().lower()

    if answer not in ("", "y", "yes"):

        print("Download cancelled.")

        return

    # --------------------------------------------------------
    # 6. Download
    # --------------------------------------------------------

    results = []

    successful = 0
    failed = 0

    for i, partition in enumerate(
        selected,
        start=1
    ):

        display_name = partition.get(
            "display_name",
            ""
        )

        url = partition["file"]

        quarter = extract_quarter(
            display_name
        )

        filename = safe_filename(url)

        quarter_dir = create_output_directory(
            quarter
        )

        destination = quarter_dir / filename

        print("\n")
        print("=" * 70)
        print(
            f"FILE {i}/{len(selected)}"
        )
        print(display_name)
        print("=" * 70)

        success = download_file(
            url=url,
            destination=destination
        )

        if success:
            successful += 1
            status = "success"
        else:
            failed += 1
            status = "failed"

        results.append({
            "display_name": display_name,
            "url": url,
            "quarter": quarter,
            "filename": filename,
            "status": status,
            "local_path": str(destination)
        })

        # Save manifest after every file so that
        # progress is not lost if the script stops.
        save_manifest(
            selected_partitions=selected,
            results=results,
            index_last_updated=data.get(
                "meta",
                {}
            ).get(
                "last_updated",
                ""
            )
        )

    # --------------------------------------------------------
    # 7. Final report
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)

    print(
        f"Successful: {successful:,}"
    )

    print(
        f"Failed:     {failed:,}"
    )

    print(
        f"Total:      {len(selected):,}"
    )

    print(
        f"\nFiles are located at:\n{OUTPUT_DIR}"
    )

    if failed > 0:

        print(
            "\nSome files failed."
            "\nRun the script again to retry them."
        )


if __name__ == "__main__":
    main()