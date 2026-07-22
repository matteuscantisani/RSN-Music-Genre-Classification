import urllib.request
from pathlib import Path

ZIP_DIR = Path("data_loader/data/zip")

HF_BASE = (
    "https://huggingface.co/datasets/"
    "matteuscantisani/rsn-music-genre-classification/"
    "resolve/main"
)

ZIP_URLS = {
    "Forró.zip": f"{HF_BASE}/Forró.zip",
    "Funk.zip": f"{HF_BASE}/Funk.zip",
    "Samba_Pagode.zip": f"{HF_BASE}/Samba_Pagode.zip",
}


def download_all():
    ZIP_DIR.mkdir(parents=True, exist_ok=True)

    for name, url in ZIP_URLS.items():
        dest = ZIP_DIR / name
        if dest.exists():
            print(f"Já existe: {dest}")
            continue
        print(f"Baixando {name} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"  OK")


if __name__ == "__main__":
    download_all()
