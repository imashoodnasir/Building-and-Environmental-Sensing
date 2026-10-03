from pathlib import Path
import hashlib
import urllib.request
import sys


# ============================================================
# Configuration
# ============================================================

URL = (
    "https://zenodo.org/records/3887306/files/"
    "buds-lab%2Fbuilding-data-genome-project-2-v1.0.zip"
    "?download=1"
)

EXPECTED_MD5 = "44393dc4cf61e84dec105e955368c890"

ROOT = Path(__file__).resolve().parents[1]

OUT_DIR = ROOT / "data" / "raw" / "bdg2"

OUT_FILE = (
    OUT_DIR /
    "building-data-genome-project-2-v1.0.zip"
)


# ============================================================
# MD5
# ============================================================

def calculate_md5(path, chunk_size=1024 * 1024):

    md5 = hashlib.md5()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(chunk_size)

            if not chunk:
                break

            md5.update(chunk)

    return md5.hexdigest()


# ============================================================
# Download progress
# ============================================================

def progress(block_num, block_size, total_size):

    downloaded = block_num * block_size

    if total_size > 0:

        percent = min(
            100,
            downloaded * 100 / total_size
        )

        mb_downloaded = downloaded / 1024**2
        mb_total = total_size / 1024**2

        print(
            f"\rDownloading: "
            f"{mb_downloaded:,.1f} / "
            f"{mb_total:,.1f} MB "
            f"({percent:.1f}%)",
            end=""
        )


# ============================================================
# Main
# ============================================================

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("=" * 72)
print("BDG2 DATA ACQUISITION")
print("=" * 72)

print()
print("Source:")
print(URL)

print()
print("Destination:")
print(OUT_FILE)

print()


# ------------------------------------------------------------
# Existing file
# ------------------------------------------------------------

if OUT_FILE.exists():

    print("ZIP already exists.")
    print("Checking integrity...")

else:

    print("Downloading official BDG2 v1.0 archive...")
    print()

    try:

        urllib.request.urlretrieve(
            URL,
            OUT_FILE,
            reporthook=progress
        )

        print()

    except Exception as e:

        print()
        print("DOWNLOAD FAILED")
        print(e)

        if OUT_FILE.exists():
            OUT_FILE.unlink()

        sys.exit(1)


# ------------------------------------------------------------
# Integrity verification
# ------------------------------------------------------------

print()
print("Calculating MD5...")

actual_md5 = calculate_md5(
    OUT_FILE
)

print()
print("Expected MD5:", EXPECTED_MD5)
print("Actual MD5:  ", actual_md5)


if actual_md5.lower() != EXPECTED_MD5.lower():

    print()
    print("ERROR: MD5 CHECK FAILED.")
    print(
        "The downloaded archive does not match "
        "the official Zenodo file."
    )

    sys.exit(1)


print()
print("MD5 CHECK: PASSED")

size_mb = (
    OUT_FILE.stat().st_size /
    1024**2
)

print(
    f"Archive size: {size_mb:.1f} MB"
)

print()
print("=" * 72)
print("BDG2 DOWNLOAD COMPLETE")
print("=" * 72)

print()
print(
    "Do not extract or modify the archive yet."
)