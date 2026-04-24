import os
import tarfile
import urllib.request
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

ZENODO_URL = "https://zenodo.org/records/14053985/files/flyingarucov2.tar.gz"
OUT_DIR = "data/raw/"
FOLDER = os.path.join(OUT_DIR, "flyingarucov2")

def download_zenodo_data(url: str = ZENODO_URL, output_dir: str = OUT_DIR) -> None:
    """
    Downloads and extracts the Flying ArUco v2 dataset from Zenodo (.tar.gz).

    Skips the download if the target folder already exists and is non-empty.

    Args:
        url:        Direct download URL for the .tar.gz archive.
        output_dir: Directory where the archive will be extracted.
    """
    os.makedirs(output_dir, exist_ok=True)

    check_folder = os.path.join(output_dir, "flyingarucov2")
    if os.path.exists(check_folder) and os.listdir(check_folder):
        print(f"[*] Zenodo data already exists at '{check_folder}'. Skipping download.")
        return

    tar_path = os.path.join(output_dir, "flyingarucov2.tar.gz")

    print("[*] Downloading Flying ArUco v2 from Zenodo (this may take a while)...")
    try:
        urllib.request.urlretrieve(url, tar_path)

        print(f"[*] Extracting dataset to '{output_dir}'...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=output_dir)

        os.remove(tar_path)
        print(f"[*] Zenodo data ready at: '{check_folder}'")

    except Exception as e:
        print(f"[!] Error downloading or extracting Zenodo data: {e}")

def download_kaggle_competition(
    competition: str = 'aruco-detection-challenge', output_dir: str = OUT_DIR
) -> None:
    """
    Downloads files from a Kaggle competition using the Kaggle API.
    Requires accepting the competition rules on Kaggle first.
    Args:
        competition: Competition identifier, e.g. 'titanic' or 'detect-aruco-markers'.
        output_dir:  Directory to download and extract the files into.
    """
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(output_dir) and os.listdir(output_dir):
        print(
            f"[*] Competition data already exists at '{output_dir}'. Skipping download."
        )
        return
    api = KaggleApi()
    api.authenticate()  # reads ~/.kaggle/kaggle.json
    print(f"[*] Downloading competition '{competition}' from Kaggle...")
    api.competition_download_files(competition, path=output_dir, quiet=False)
    # Unzip the downloaded archive
    zip_path = os.path.join(output_dir, f"{competition}.zip")
    if os.path.exists(zip_path):
        print(f"[*] Extracting to '{output_dir}'...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(output_dir)
        os.remove(zip_path)
    print(f"[*] Competition data ready at: '{output_dir}'")

def run_download_pipeline(from_kaggle: bool = False) -> None:
    if from_kaggle:
        """Runs the Kaggle data ingestion pipeline."""
        print("\n" + "=" * 50)
        print("STARTING DATA INGESTION PIPELINE (Source: KAGGLE)")
        download_kaggle_competition()
        print("=" * 50)
    else:
        """Runs the Zenodo data ingestion pipeline."""
        print("\n" + "=" * 50)
        print("STARTING DATA INGESTION PIPELINE (Source: ZENODO)")
        download_zenodo_data()
        print("=" * 50)


if __name__ == "__main__":
    run_download_pipeline()