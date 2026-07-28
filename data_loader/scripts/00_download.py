import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

ZIP_DIR = Path("data_loader/data/zip")
REPO_ID = "matteuscantisani/rsn-music-genre-classification"
ZIP_FILES = ["Forró.zip", "Funk.zip", "Samba_Pagode.zip"]


def _is_valid_zip(path: Path) -> bool:
    return zipfile.is_zipfile(path)


def download_all():
    ZIP_DIR.mkdir(parents=True, exist_ok=True)

    for name in ZIP_FILES:
        dest = ZIP_DIR / name

        if dest.exists():
            if _is_valid_zip(dest):
                print(f"Já existe: {dest}")
                continue
            print(f"Corrompido, removendo: {dest}")
            dest.unlink()

        print(f"Baixando {name} ...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=name,
            repo_type="dataset",
            local_dir=ZIP_DIR,
            local_dir_use_symlinks=False,
        )

        if not _is_valid_zip(dest):
            dest.unlink(missing_ok=True)
            raise RuntimeError(f"Download de {name} gerou um arquivo zip inválido.")

        print(f"  OK")


if __name__ == "__main__":
    download_all()
