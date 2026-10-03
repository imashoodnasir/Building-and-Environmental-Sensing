from pathlib import Path
from zipfile import ZipFile
import pandas as pd
import json
import os


ROOT = Path(__file__).resolve().parents[1]

ZIP_FILE = (
    ROOT
    / "data"
    / "raw"
    / "bdg2"
    / "building-data-genome-project-2-v1.0.zip"
)

OUTPUT_DIR = ROOT / "data" / "metadata"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 90)
print("BDG2 ARCHIVE AUDIT")
print("=" * 90)


if not ZIP_FILE.exists():
    raise FileNotFoundError(ZIP_FILE)


# ============================================================
# 1. LIST ARCHIVE CONTENTS
# ============================================================

with ZipFile(ZIP_FILE, "r") as z:

    members = [
        x for x in z.infolist()
        if not x.is_dir()
    ]

    records = []

    for x in members:

        path = Path(x.filename)

        records.append({
            "path": x.filename,
            "filename": path.name,
            "extension": path.suffix.lower(),
            "compressed_mb": x.compress_size / 1024**2,
            "uncompressed_mb": x.file_size / 1024**2,
        })


inventory = pd.DataFrame(records)

inventory.to_csv(
    OUTPUT_DIR / "bdg2_archive_inventory.csv",
    index=False
)


print()
print("FILES IN ARCHIVE:", len(inventory))

print()
print(
    inventory[
        [
            "path",
            "extension",
            "uncompressed_mb"
        ]
    ].to_string(index=False)
)


# ============================================================
# 2. SEARCH FOR RELEVANT FILE NAMES
# ============================================================

keywords = [
    "meter",
    "weather",
    "metadata",
    "temperature",
    "humidity",
    "pressure",
    "energy",
    "electric",
    "gas",
    "steam",
    "water",
    "air",
]


print()
print("=" * 90)
print("POTENTIALLY RELEVANT FILES")
print("=" * 90)


relevant = inventory[
    inventory["path"]
    .str.lower()
    .apply(
        lambda x: any(
            keyword in x
            for keyword in keywords
        )
    )
]


if len(relevant):

    print(
        relevant[
            [
                "path",
                "uncompressed_mb"
            ]
        ].to_string(index=False)
    )

else:

    print("No filenames matched the keyword search.")


relevant.to_csv(
    OUTPUT_DIR / "bdg2_relevant_files.csv",
    index=False
)


# ============================================================
# 3. INSPECT CSV HEADERS
# ============================================================

print()
print("=" * 90)
print("CSV SCHEMA AUDIT")
print("=" * 90)


schema_records = []


with ZipFile(ZIP_FILE, "r") as z:

    csv_files = [
        x.filename
        for x in members
        if x.filename.lower().endswith(".csv")
    ]

    print()
    print("CSV files:", len(csv_files))

    for filename in csv_files:

        print()
        print("-" * 90)
        print(filename)

        try:

            with z.open(filename) as f:

                sample = pd.read_csv(
                    f,
                    nrows=5,
                    low_memory=False
                )

            columns = [
                str(c)
                for c in sample.columns
            ]

            print("Columns:")
            print(columns)

            print()
            print("Sample shape:", sample.shape)

            schema_records.append({
                "file": filename,
                "n_columns": len(columns),
                "columns": " | ".join(columns),
                "status": "OK",
            })

        except Exception as e:

            print("Could not parse:", e)

            schema_records.append({
                "file": filename,
                "n_columns": None,
                "columns": None,
                "status": str(e),
            })


schema = pd.DataFrame(schema_records)

schema.to_csv(
    OUTPUT_DIR / "bdg2_csv_schema.csv",
    index=False
)


# ============================================================
# 4. SEARCH COLUMN NAMES
# ============================================================

print()
print("=" * 90)
print("VARIABLE SEARCH IN CSV COLUMN NAMES")
print("=" * 90)


variable_terms = {
    "temperature": [
        "temperature",
        "temp"
    ],

    "relative_humidity": [
        "humidity",
        "relative_humidity",
        "rh"
    ],

    "pressure": [
        "pressure",
        "barometric",
        "atmospheric"
    ],

    "energy": [
        "electricity",
        "electric",
        "energy",
        "gas",
        "steam",
        "chilledwater",
        "hotwater"
    ],

    "pm25": [
        "pm25",
        "pm2.5",
        "particulate"
    ]
}


matches = []


for _, row in schema.iterrows():

    if pd.isna(row["columns"]):
        continue

    columns = [
        x.strip()
        for x in row["columns"].split("|")
    ]

    for column in columns:

        c = column.lower()

        for variable, terms in variable_terms.items():

            if any(
                term in c
                for term in terms
            ):

                matches.append({
                    "variable": variable,
                    "file": row["file"],
                    "column": column
                })


matches_df = pd.DataFrame(matches)


print()

if len(matches_df):

    print(
        matches_df.to_string(
            index=False
        )
    )

else:

    print(
        "No obvious variable names were identified "
        "from CSV headers."
    )


matches_df.to_csv(
    OUTPUT_DIR / "bdg2_variable_matches.csv",
    index=False
)


# ============================================================
# 5. CREATE MACHINE-READABLE AUDIT
# ============================================================

summary = {

    "archive": ZIP_FILE.name,

    "number_of_files": int(
        len(inventory)
    ),

    "number_of_csv_files": int(
        len(schema)
    ),

    "extensions": (
        inventory["extension"]
        .value_counts()
        .to_dict()
    ),

    "candidate_variables": {}
}


for variable in variable_terms:

    if len(matches_df):

        subset = matches_df[
            matches_df["variable"] == variable
        ]

        summary["candidate_variables"][variable] = (
            subset[
                ["file", "column"]
            ]
            .to_dict("records")
        )

    else:

        summary["candidate_variables"][variable] = []


with open(
    OUTPUT_DIR / "bdg2_audit_summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 90)
print("BDG2 ARCHIVE AUDIT COMPLETE")
print("=" * 90)

print()
print("Generated:")
print(
    "  data/metadata/bdg2_archive_inventory.csv"
)
print(
    "  data/metadata/bdg2_relevant_files.csv"
)
print(
    "  data/metadata/bdg2_csv_schema.csv"
)
print(
    "  data/metadata/bdg2_variable_matches.csv"
)
print(
    "  data/metadata/bdg2_audit_summary.json"
)

print()
print(
    "No raw archive contents were modified."
)