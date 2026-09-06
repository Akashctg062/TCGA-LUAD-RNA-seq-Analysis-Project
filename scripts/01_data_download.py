"""
01_data_download.py
====================
Download TCGA-LUAD RNA-seq gene expression counts and clinical data
from the GDC (Genomic Data Commons) REST API.

Usage:
    python scripts/01_data_download.py

Output:
    data/raw_counts.csv          - Gene expression count matrix (genes × samples)
    data/clinical_data.csv       - Clinical metadata for all cases
    data/sample_map.csv          - Mapping of file UUID → TCGA barcode → sample type

Notes:
    - Downloads STAR - Counts workflow data (unstranded counts)
    - Total download: ~600 files, ~500MB compressed
    - Runtime: 30-60 minutes depending on internet speed
    - GDC API does not require authentication for open-access data
"""

import os
import io
import json
import gzip
import time
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"
GDC_CASES_ENDPOINT = "https://api.gdc.cancer.gov/cases"

BATCH_SIZE = 50  # Number of files to download per batch request


# ──────────────────────────────────────────────
# Step 1: Query GDC for TCGA-LUAD RNA-seq files
# ──────────────────────────────────────────────
def query_rnaseq_files():
    """Query GDC API for all TCGA-LUAD RNA-seq HTSeq count files."""
    print("=" * 60)
    print("Step 1: Querying GDC for TCGA-LUAD RNA-seq files...")
    print("=" * 60)

    filters = {
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "cases.project.project_id", "value": "TCGA-LUAD"}},
            {"op": "=", "content": {"field": "data_category", "value": "Transcriptome Profiling"}},
            {"op": "=", "content": {"field": "data_type", "value": "Gene Expression Quantification"}},
            {"op": "=", "content": {"field": "analysis.workflow_type", "value": "STAR - Counts"}},
            {"op": "=", "content": {"field": "data_format", "value": "TSV"}},
        ]
    }

    fields = [
        "file_id",
        "file_name",
        "file_size",
        "cases.case_id",
        "cases.submitter_id",
        "cases.samples.sample_type",
        "cases.samples.submitter_id",
        "cases.samples.portions.analytes.aliquots.submitter_id",
    ]

    params = {
        "filters": json.dumps(filters),
        "fields": ",".join(fields),
        "format": "JSON",
        "size": 1000,
    }

    response = requests.get(GDC_FILES_ENDPOINT, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()

    hits = data["data"]["hits"]
    total = data["data"]["pagination"]["total"]
    print(f"  Found {total} RNA-seq files")

    # Build file metadata table
    records = []
    for hit in hits:
        file_id = hit["file_id"]
        file_name = hit["file_name"]
        file_size = hit["file_size"]

        for case in hit.get("cases", []):
            case_id = case.get("case_id", "")
            submitter_id = case.get("submitter_id", "")  # e.g., TCGA-XX-XXXX

            for sample in case.get("samples", []):
                sample_type = sample.get("sample_type", "")
                sample_submitter = sample.get("submitter_id", "")  # e.g., TCGA-XX-XXXX-01A

                records.append({
                    "file_id": file_id,
                    "file_name": file_name,
                    "file_size": file_size,
                    "case_id": case_id,
                    "submitter_id": submitter_id,
                    "sample_submitter_id": sample_submitter,
                    "sample_type": sample_type,
                })

    file_df = pd.DataFrame(records)
    print(f"  Samples breakdown:")
    print(f"    {file_df['sample_type'].value_counts().to_string()}")
    print()

    return file_df


# ──────────────────────────────────────────────
# Step 2: Download count files in batches
# ──────────────────────────────────────────────
def download_count_files(file_df):
    """Download gene expression count files from GDC in batches."""
    print("=" * 60)
    print("Step 2: Downloading count files from GDC...")
    print("=" * 60)

    file_ids = file_df["file_id"].unique().tolist()
    total_files = len(file_ids)
    print(f"  Downloading {total_files} files in batches of {BATCH_SIZE}...")

    all_counts = {}  # file_id → DataFrame of counts
    failed_files = []

    for batch_start in tqdm(range(0, total_files, BATCH_SIZE), desc="  Batches"):
        batch_ids = file_ids[batch_start : batch_start + BATCH_SIZE]

        # GDC bulk download endpoint
        payload = {"ids": batch_ids}
        try:
            response = requests.post(
                GDC_DATA_ENDPOINT,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=300,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"\n  WARNING: Batch starting at {batch_start} failed: {e}")
            failed_files.extend(batch_ids)
            continue

        # If single file, response is the file directly
        # If multiple files, response is a tar.gz archive
        content_type = response.headers.get("Content-Type", "")

        if "application/x-tar" in content_type or len(batch_ids) > 1:
            # Extract tar archive
            import tarfile
            tar_bytes = io.BytesIO(response.content)
            try:
                with tarfile.open(fileobj=tar_bytes, mode="r:gz") as tar:
                    for member in tar.getmembers():
                        if member.name.endswith(".tsv") and not member.name.startswith("MANIFEST"):
                            f = tar.extractfile(member)
                            if f is not None:
                                content = f.read().decode("utf-8")
                                # Extract file_id from path (format: UUID/filename.tsv)
                                parts = member.name.split("/")
                                fid = parts[0] if len(parts) > 1 else member.name
                                all_counts[fid] = parse_star_counts(content)
            except Exception as e:
                print(f"\n  WARNING: Error extracting batch: {e}")
                failed_files.extend(batch_ids)
        else:
            # Single file
            content = response.content.decode("utf-8")
            all_counts[batch_ids[0]] = parse_star_counts(content)

        # Brief pause to be respectful to GDC servers
        time.sleep(0.5)

    print(f"\n  Successfully downloaded: {len(all_counts)} / {total_files} files")
    if failed_files:
        print(f"  Failed files: {len(failed_files)} (will retry individually)")
        all_counts = retry_failed_downloads(failed_files, all_counts)

    return all_counts


def parse_star_counts(tsv_content):
    """Parse a STAR counts TSV file and return gene_id → unstranded count mapping."""
    df = pd.read_csv(io.StringIO(tsv_content), sep="\t", comment="#")

    # STAR counts files have columns: gene_id, gene_name, gene_type, unstranded, ...
    # We want gene_id (Ensembl ID) and unstranded counts
    # Skip the first 4 rows which are summary statistics (N_unmapped, etc.)
    if "gene_id" in df.columns and "unstranded" in df.columns:
        df = df[~df["gene_id"].str.startswith("N_")]
        return df.set_index("gene_id")["unstranded"]
    elif "gene_id" in df.columns and "tpm_unstranded" in df.columns:
        # Fallback: some files may have different column names
        df = df[~df["gene_id"].str.startswith("N_")]
        return df.set_index("gene_id")["unstranded"]
    else:
        # Try positional parsing
        df.columns = ["gene_id", "gene_name", "gene_type", "unstranded",
                       "stranded_first", "stranded_second", "tpm_unstranded",
                       "fpkm_unstranded", "fpkm_uq_unstranded"][:len(df.columns)]
        df = df[~df["gene_id"].astype(str).str.startswith("N_")]
        return df.set_index("gene_id")["unstranded"]


def retry_failed_downloads(failed_ids, all_counts, max_retries=3):
    """Retry downloading individual failed files."""
    for fid in tqdm(failed_ids, desc="  Retrying failed"):
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    f"{GDC_DATA_ENDPOINT}/{fid}",
                    timeout=120,
                )
                response.raise_for_status()

                # Check if gzipped
                if response.content[:2] == b'\x1f\x8b':
                    content = gzip.decompress(response.content).decode("utf-8")
                else:
                    content = response.content.decode("utf-8")

                all_counts[fid] = parse_star_counts(content)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"\n  FAILED permanently: {fid} - {e}")
                time.sleep(2)

    return all_counts


# ──────────────────────────────────────────────
# Step 3: Build count matrix
# ──────────────────────────────────────────────
def build_count_matrix(all_counts, file_df):
    """Merge individual count series into a genes × samples matrix."""
    print("=" * 60)
    print("Step 3: Building count matrix...")
    print("=" * 60)

    # Map file_id to a clean sample label (TCGA barcode)
    fid_to_barcode = {}
    for _, row in file_df.drop_duplicates("file_id").iterrows():
        barcode = row["sample_submitter_id"]
        if not barcode or pd.isna(barcode):
            barcode = row["submitter_id"]
        fid_to_barcode[row["file_id"]] = barcode

    # Build matrix
    count_series = {}
    for fid, counts in all_counts.items():
        label = fid_to_barcode.get(fid, fid)
        count_series[label] = counts

    count_matrix = pd.DataFrame(count_series)
    count_matrix.index.name = "gene_id"

    # Remove version suffix from Ensembl IDs (ENSG00000000003.15 → ENSG00000000003)
    count_matrix.index = count_matrix.index.str.split(".").str[0]

    # Drop any duplicate gene IDs (keep first)
    count_matrix = count_matrix[~count_matrix.index.duplicated(keep="first")]

    # Convert to integer
    count_matrix = count_matrix.fillna(0).astype(int)

    print(f"  Count matrix shape: {count_matrix.shape[0]} genes × {count_matrix.shape[1]} samples")
    print(f"  Memory usage: {count_matrix.memory_usage().sum() / 1e6:.1f} MB")

    return count_matrix


# ──────────────────────────────────────────────
# Step 4: Get gene name mapping
# ──────────────────────────────────────────────
def get_gene_name_mapping(all_counts, file_df):
    """Extract gene_id → gene_name mapping from one of the downloaded files."""
    print("  Extracting gene ID → gene name mapping...")

    # Re-download one file to get the full TSV with gene names
    fid = list(all_counts.keys())[0]
    try:
        response = requests.get(f"{GDC_DATA_ENDPOINT}/{fid}", timeout=120)
        if response.content[:2] == b'\x1f\x8b':
            content = gzip.decompress(response.content).decode("utf-8")
        else:
            content = response.content.decode("utf-8")

        df = pd.read_csv(io.StringIO(content), sep="\t", comment="#")
        if "gene_id" in df.columns and "gene_name" in df.columns:
            df = df[~df["gene_id"].str.startswith("N_")]
            df["gene_id_clean"] = df["gene_id"].str.split(".").str[0]
            mapping = df.set_index("gene_id_clean")[["gene_name", "gene_type"]]
            mapping = mapping[~mapping.index.duplicated(keep="first")]
            print(f"  Mapped {len(mapping)} gene IDs to gene names")
            return mapping
    except Exception as e:
        print(f"  WARNING: Could not extract gene mapping: {e}")

    return None


# ──────────────────────────────────────────────
# Step 5: Download clinical data
# ──────────────────────────────────────────────
def download_clinical_data():
    """Download clinical metadata for all TCGA-LUAD cases."""
    print("=" * 60)
    print("Step 4: Downloading clinical data...")
    print("=" * 60)

    filters = {
        "op": "=",
        "content": {
            "field": "project.project_id",
            "value": "TCGA-LUAD"
        }
    }

    fields = [
        "submitter_id",
        "case_id",
        "diagnoses.age_at_diagnosis",
        "diagnoses.tumor_stage",
        "diagnoses.ajcc_pathologic_stage",
        "diagnoses.primary_diagnosis",
        "diagnoses.vital_status",
        "diagnoses.days_to_death",
        "diagnoses.days_to_last_follow_up",
        "demographic.gender",
        "demographic.race",
        "demographic.ethnicity",
        "demographic.vital_status",
        "demographic.days_to_death",
        "exposures.cigarettes_per_day",
        "exposures.years_smoked",
        "exposures.tobacco_smoking_status",
    ]

    params = {
        "filters": json.dumps(filters),
        "fields": ",".join(fields),
        "format": "JSON",
        "size": 1000,
    }

    response = requests.get(GDC_CASES_ENDPOINT, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()

    cases = data["data"]["hits"]
    print(f"  Found {len(cases)} cases")

    records = []
    for case in cases:
        record = {
            "case_id": case.get("case_id", ""),
            "submitter_id": case.get("submitter_id", ""),
        }

        # Demographics
        demo = case.get("demographic", {}) or {}
        record["gender"] = demo.get("gender", "")
        record["race"] = demo.get("race", "")
        record["ethnicity"] = demo.get("ethnicity", "")
        record["vital_status"] = demo.get("vital_status", "")
        record["days_to_death"] = demo.get("days_to_death", None)

        # Diagnoses (take first)
        diags = case.get("diagnoses", []) or []
        if diags:
            diag = diags[0]
            record["age_at_diagnosis"] = diag.get("age_at_diagnosis", None)
            record["tumor_stage"] = diag.get("ajcc_pathologic_stage",
                                              diag.get("tumor_stage", ""))
            record["primary_diagnosis"] = diag.get("primary_diagnosis", "")
            record["vital_status_diag"] = diag.get("vital_status", "")
            record["days_to_death_diag"] = diag.get("days_to_death", None)
            record["days_to_last_follow_up"] = diag.get("days_to_last_follow_up", None)

        # Exposures (take first)
        exposures = case.get("exposures", []) or []
        if exposures:
            exp = exposures[0]
            record["cigarettes_per_day"] = exp.get("cigarettes_per_day", None)
            record["years_smoked"] = exp.get("years_smoked", None)
            record["tobacco_smoking_status"] = exp.get("tobacco_smoking_status", None)

        records.append(record)

    clinical_df = pd.DataFrame(records)

    # Compute overall survival time and event
    clinical_df["os_time_days"] = clinical_df["days_to_death"].combine_first(
        clinical_df["days_to_death_diag"]
    ).combine_first(
        clinical_df["days_to_last_follow_up"]
    )
    clinical_df["os_event"] = clinical_df["vital_status"].apply(
        lambda x: 1 if str(x).lower() == "dead" else 0
    )

    # Convert age from days to years
    if clinical_df["age_at_diagnosis"].notna().any():
        clinical_df["age_years"] = (clinical_df["age_at_diagnosis"] / 365.25).round(1)

    print(f"  Clinical data shape: {clinical_df.shape}")
    print(f"  Vital status: {clinical_df['vital_status'].value_counts().to_string()}")
    print(f"  Stages: {clinical_df['tumor_stage'].value_counts().head().to_string()}")

    return clinical_df


# ──────────────────────────────────────────────
# Step 6: Create sample type mapping
# ──────────────────────────────────────────────
def create_sample_map(file_df):
    """Create a clean mapping of sample barcode → sample type (Tumor/Normal)."""
    sample_map = file_df[["file_id", "submitter_id", "sample_submitter_id", "sample_type"]].copy()
    sample_map = sample_map.drop_duplicates("file_id")

    # Simplify sample types
    sample_map["condition"] = sample_map["sample_type"].apply(
        lambda x: "Normal" if "Normal" in str(x) else "Tumor"
    )

    print(f"\n  Sample type summary:")
    print(f"    {sample_map['condition'].value_counts().to_string()}")

    return sample_map


# ──────────────────────────────────────────────
# Main execution
# ──────────────────────────────────────────────
def main():
    start_time = time.time()
    print("\n" + "=" * 60)
    print("  TCGA-LUAD RNA-seq Data Download Pipeline")
    print("=" * 60 + "\n")

    # Step 1: Query files
    file_df = query_rnaseq_files()

    # Step 2: Download count files
    all_counts = download_count_files(file_df)

    if len(all_counts) == 0:
        print("\nERROR: No files were successfully downloaded. Check your internet connection.")
        return

    # Step 3: Build count matrix
    count_matrix = build_count_matrix(all_counts, file_df)

    # Get gene name mapping
    gene_mapping = get_gene_name_mapping(all_counts, file_df)

    # Step 4: Download clinical data
    clinical_df = download_clinical_data()

    # Step 5: Create sample map
    sample_map = create_sample_map(file_df)

    # ── Save all outputs ──
    print("\n" + "=" * 60)
    print("Saving outputs...")
    print("=" * 60)

    count_matrix.to_csv(DATA_DIR / "raw_counts.csv")
    print(f"  Saved: data/raw_counts.csv ({count_matrix.shape[0]} genes × {count_matrix.shape[1]} samples)")

    clinical_df.to_csv(DATA_DIR / "clinical_data.csv", index=False)
    print(f"  Saved: data/clinical_data.csv ({len(clinical_df)} cases)")

    sample_map.to_csv(DATA_DIR / "sample_map.csv", index=False)
    print(f"  Saved: data/sample_map.csv ({len(sample_map)} samples)")

    if gene_mapping is not None:
        gene_mapping.to_csv(DATA_DIR / "gene_id_mapping.csv")
        print(f"  Saved: data/gene_id_mapping.csv ({len(gene_mapping)} genes)")

    elapsed = time.time() - start_time
    print(f"\nDone! Total time: {elapsed / 60:.1f} minutes")
    print(f"Next step: Run python scripts/02_preprocessing_qc.py")


if __name__ == "__main__":
    main()
