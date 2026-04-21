import os
import tarfile
import urllib.request

ZENODO_URL = "https://zenodo.org/records/14053985/files/flyingarucov2.tar.gz"
ZENODO_OUT_DIR = "data/raw/"
ZENODO_FOLDER = os.path.join(ZENODO_OUT_DIR, "flyingarucov2")


def download_zenodo_data(url: str = ZENODO_URL, output_dir: str = ZENODO_OUT_DIR) -> None:
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


def run_download_pipeline() -> None:
    """Runs the Zenodo data ingestion pipeline."""
    print("\n" + "=" * 50)
    print("STARTING DATA INGESTION PIPELINE (Source: ZENODO)")
    download_zenodo_data()
    print("=" * 50)


if __name__ == "__main__":
    run_download_pipeline()