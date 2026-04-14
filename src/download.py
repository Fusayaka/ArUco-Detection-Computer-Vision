import os
import zipfile
import tarfile
import urllib.request
import subprocess

def download_kaggle_data(competition_name, output_dir):
    """
    Downloads and extracts dataset from a Kaggle competition.
    Used for getting the clean test set.
    """
    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, f"{competition_name}.zip")

    print(f"[*] Connecting to Kaggle to download '{competition_name}'...")
    
    try:
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", competition_name, "-p", output_dir],
            check=True
        )
    except subprocess.CalledProcessError:
        print("[!] Error: Failed to download from Kaggle.")
        print("    Ensure your 'kaggle.json' is configured correctly.")
        os.rmdir(output_dir)
        return

    if os.path.exists(zip_path):
        print(f"[*] Extracting Kaggle dataset to {output_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        os.remove(zip_path)
        print(f"Kaggle data ready at: {output_dir}")
    else:
        print("[!] Kaggle zip file not found. Assuming data is already present.")


def download_zenodo_data(url, output_dir):
    """
    Downloads and extracts the Flying ArUco v2 dataset from Zenodo (.tar.gz).
    Used for getting the raw training data (images + JSONs).
    """
    os.makedirs(output_dir, exist_ok=True)
    tar_path = os.path.join(output_dir, "flyingarucov2.tar.gz")

    check_folder = os.path.join(output_dir, "flyingarucov2")
    if os.path.exists(check_folder) and len(os.listdir(check_folder)) > 0:
        print(f"[*] Zenodo data already exists at {check_folder}. Skipping download.")
        return

    print(f"[*] Downloading Flying ArUco v2 from Zenodo (This might take a while)...")
    try:
        urllib.request.urlretrieve(url, tar_path)
        
        print(f"[*] Extracting .tar.gz dataset to {output_dir}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=output_dir)
            
        os.remove(tar_path)
        print(f"Zenodo data ready at: {check_folder}")
        
    except Exception as e:
        print(f"[!] Error downloading or extracting Zenodo data: {e}")

def run_download_pipeline(source="both"):
    """
    Triggers data ingestion tasks based on the selected source.
    Args:
        source (str): 'both', 'kaggle', or 'zenodo'.
    """
    print("\n" + "="*50)
    print(f"STARTING DATA INGESTION PIPELINE (Source: {source.upper()})")
    print("="*50)

    KAGGLE_COMPETITION = "aruco-detection-challenge"
    KAGGLE_OUT_DIR = "data/raw/aruco_data"
    
    ZENODO_URL = "https://zenodo.org/records/14053985/files/flyingarucov2.tar.gz"
    ZENODO_OUT_DIR = "data/raw/"

    if source in ["kaggle", "both"]:
        download_kaggle_data(KAGGLE_COMPETITION, KAGGLE_OUT_DIR)

    if source in ["zenodo", "both"]:
        download_zenodo_data(ZENODO_URL, ZENODO_OUT_DIR)

if __name__ == "__main__":
    run_download_pipeline(source="both")