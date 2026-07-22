import urllib.request
import urllib.parse
from pathlib import Path

ZIP_DIR = Path("data_loader/data/zip")

HF_BASE = (
    "https://huggingface.co/datasets/"
    "matteuscantisani/rsn-music-genre-classification/"
    "resolve/main"
)

ZIP_FILES = ["Forró.zip", "Funk.zip", "Samba_Pagode.zip"]


def download_all():
    ZIP_DIR.mkdir(parents=True, exist_ok=True)

    for name in ZIP_FILES:
        dest = ZIP_DIR / name
        if dest.exists():
            print(f"Já existe: {dest}")
            continue
        url = f"{HF_BASE}/{urllib.parse.quote(name)}"
        print(f"Baixando {name} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"  OK")


if __name__ == "__main__":
    download_all()
