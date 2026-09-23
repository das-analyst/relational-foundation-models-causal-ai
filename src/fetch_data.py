"""
Data Fetcher for 130-US Hospitals Clinical Outcomes & Readmission Dataset.
Source: UCI Machine Learning Repository / National Institutes of Health.
"""

import os
import urllib.request
import zipfile
import sys

DATA_URL = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"

def fetch_and_extract_data(target_dir: str = "data/raw"):
    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, "dataset_diabetes.zip")
    extracted_csv = os.path.join(target_dir, "diabetic_data.csv")

    if os.path.exists(extracted_csv):
        print(f"[OK] Dataset already present at: {extracted_csv}")
        return extracted_csv

    print(f"[*] Downloading 130-US Hospitals clinical dataset from:\n    {DATA_URL}")
    
    def reporthook(blocknum, blocksize, totalsize):
        readsofar = blocknum * blocksize
        if totalsize > 0:
            percent = readsofar * 1e2 / totalsize
            s = f"\r--> {readsofar / (1024*1024):.2f} MB / {totalsize / (1024*1024):.2f} MB [{percent:.1f}%]"
            sys.stdout.write(s)
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(DATA_URL, zip_path, reporthook=reporthook)
        print("\n[OK] Download complete.")
    except Exception as e:
        print(f"\n[!] Download failed: {e}")
        raise

    print(f"[*] Extracting files to: {target_dir}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
        print(f"[OK] Extracted files: {zip_ref.namelist()}")

    # Clean up zip
    if os.path.exists(zip_path):
        os.remove(zip_path)

    return extracted_csv

if __name__ == "__main__":
    fetch_and_extract_data()
