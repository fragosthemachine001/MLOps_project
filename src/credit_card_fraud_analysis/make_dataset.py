import subprocess
from pathlib import Path

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

# Create folders if they don't exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_kaggle_dataset(dataset: str, filename: str) -> Path:
    """
    Fetch dataset from Kaggle if it does not already exist.

    Parameters
    ----------
    dataset : str
        Kaggle dataset identifier (e.g. 'zynicide/wine-reviews')
    filename : str
        Expected CSV filename inside the dataset

    Returns
    -------
    Path
        Path to the downloaded CSV
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RAW_DATA_DIR / filename

    if output_file.exists():
        print(f"[INFO] Dataset already exists: {output_file}")
        return output_file

    print("[INFO] Downloading dataset from Kaggle...")
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            dataset,
            "-p",
            str(RAW_DATA_DIR),
            "--unzip",
        ],
        check=True,
    )

    if not output_file.exists():
        raise FileNotFoundError(f"{filename} not found after Kaggle download")

    return output_file


if __name__ == "__main__":
    fetch_kaggle_dataset(
        dataset="mlg-ulb/creditcardfraud",
        filename="creditcard.csv",
    )
