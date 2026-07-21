import zipfile
from pathlib import Path

ZIP_DIR = Path("data_loader/data/zip")
RAW_DIR = Path("data_loader/data/raw")

def extract_all():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for zip_path in sorted(ZIP_DIR.glob("*.zip")):
        genre = zip_path.stem
        extract_to = RAW_DIR / genre
        extract_to.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)

        print(f"Extraído: {zip_path.name} -> {extract_to}")

if __name__ == "__main__":
    extract_all()
