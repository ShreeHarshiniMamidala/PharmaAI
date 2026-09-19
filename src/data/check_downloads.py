from pathlib import Path
import zipfile

# ============================================================
# PharmaAI - FAERS Download Verification
# ============================================================

# Exact location of your PharmaAI project
PROJECT_ROOT = Path(r"C:\Users\mamid\OneDrive\Desktop\PharmaAI")

# Exact FAERS data location
FAERS_DIR = PROJECT_ROOT / "data" / "raw" / "faers"

print("=" * 60)
print("PharmaAI - FAERS Download Verification")
print("=" * 60)

print("\nProject folder:")
print(PROJECT_ROOT)

print("\nFAERS data folder:")
print(FAERS_DIR)

# ============================================================
# Check FAERS folder
# ============================================================

if not FAERS_DIR.exists():
    print("\n❌ FAERS folder was not found.")
    print(f"Expected location:\n{FAERS_DIR}")
    raise SystemExit

print("\n✅ FAERS folder found!")

# ============================================================
# Find quarter folders
# ============================================================

quarter_folders = sorted(
    folder for folder in FAERS_DIR.iterdir()
    if folder.is_dir()
)

if not quarter_folders:
    print("\n❌ No quarter folders found.")
    raise SystemExit

print(f"\nFound {len(quarter_folders)} quarter folders.")

# ============================================================
# Check ZIP files
# ============================================================

total_zip_files = 0
total_valid_files = 0
total_invalid_files = 0
total_size_bytes = 0

for quarter_dir in quarter_folders:

    zip_files = sorted(quarter_dir.glob("*.zip"))

    quarter_size = sum(
        file.stat().st_size for file in zip_files
    )

    valid_files = 0
    invalid_files = []

    for file in zip_files:

        try:
            with zipfile.ZipFile(file, "r") as archive:

                # Test ZIP integrity
                bad_file = archive.testzip()

                if bad_file is None:
                    valid_files += 1
                else:
                    invalid_files.append(file.name)

        except zipfile.BadZipFile:
            invalid_files.append(file.name)

    total_zip_files += len(zip_files)
    total_valid_files += valid_files
    total_invalid_files += len(invalid_files)
    total_size_bytes += quarter_size

    size_gb = quarter_size / (1024 ** 3)

    print("\n" + "-" * 60)
    print(f"Quarter: {quarter_dir.name}")
    print("-" * 60)

    print(f"ZIP files : {len(zip_files)}")
    print(f"Valid     : {valid_files}")
    print(f"Invalid   : {len(invalid_files)}")
    print(f"Size      : {size_gb:.2f} GB")

    if invalid_files:
        print("\n⚠️ Invalid files:")

        for name in invalid_files:
            print(f"   {name}")

# ============================================================
# Final summary
# ============================================================

total_size_gb = total_size_bytes / (1024 ** 3)

print("\n")
print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(f"Quarter folders : {len(quarter_folders)}")
print(f"Total ZIP files : {total_zip_files}")
print(f"Valid ZIPs      : {total_valid_files}")
print(f"Invalid ZIPs    : {total_invalid_files}")
print(f"Total size      : {total_size_gb:.2f} GB")

# ============================================================
# Final result
# ============================================================

if total_invalid_files == 0:
    print("\n✅ ALL ZIP FILES PASSED THE INTEGRITY CHECK.")
else:
    print("\n❌ SOME ZIP FILES ARE INVALID OR CORRUPTED.")
    print("Those files should be downloaded again.")

print("\nVerification complete.")