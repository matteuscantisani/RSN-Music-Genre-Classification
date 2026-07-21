import shutil
import tarfile
import urllib.request
from pathlib import Path

GTZAN_URL = (
    "https://huggingface.co/datasets/marsyas/gtzan/"
    "resolve/main/data/genres.tar.gz"
)
TAR_PATH = Path("data_loader/data/genres.tar.gz")
GTZAN_RAW = Path("data_loader/data/gtzan_raw")
DST_DIR = Path("data_loader/data/final_dataset")

GENRES_EXPECTED = 10
FILES_PER_GENRE = 100


def download():
    if TAR_PATH.exists():
        print(f"Arquivo já existe: {TAR_PATH}")
        return
    print(f"Baixando GTZan de {GTZAN_URL} ...")
    urllib.request.urlretrieve(GTZAN_URL, TAR_PATH)
    print("Download concluído.")


def extract():
    if GTZAN_RAW.exists():
        shutil.rmtree(GTZAN_RAW)
    GTZAN_RAW.mkdir(parents=True)
    print("Extraindo genres.tar.gz ...")
    with tarfile.open(TAR_PATH, "r:gz") as tar:
        tar.extractall(GTZAN_RAW)
    print("Extração concluída.")


def copy_files():
    DST_DIR.mkdir(parents=True, exist_ok=True)

    src_genres = GTZAN_RAW / "genres"
    if not src_genres.is_dir():
        # Try without the 'genres' wrapper
        src_genres = GTZAN_RAW

    genre_dirs = sorted(
        d for d in src_genres.iterdir() if d.is_dir() and not d.name.startswith("._")
    )
    counts = {}

    for genre_dir in genre_dirs:
        genre = genre_dir.name
        wav_files = sorted(
            f for f in genre_dir.iterdir()
            if f.suffix == ".wav" and not f.name.startswith("._")
        )
        if not wav_files:
            continue

        n = 0
        for wav_path in wav_files:
            dst_path = DST_DIR / wav_path.name
            shutil.copy2(wav_path, dst_path)
            n += 1

        counts[genre] = n
        print(f"  {genre}: {n} arquivos copiados")

    return counts


def verify(counts):
    total = sum(counts.values())
    n_genres = len(counts)

    if n_genres != GENRES_EXPECTED:
        print(
            f"AVISO: esperados {GENRES_EXPECTED} gêneros, "
            f"encontrados {n_genres}"
        )

    all_ok = True
    for genre, n in sorted(counts.items()):
        if n != FILES_PER_GENRE:
            print(
                f"AVISO: {genre} → {n} arquivos "
                f"(esperado {FILES_PER_GENRE})"
            )
            all_ok = False

    if total != GENRES_EXPECTED * FILES_PER_GENRE:
        print(
            f"AVISO: total de {total} arquivos "
            f"(esperado {GENRES_EXPECTED * FILES_PER_GENRE})"
        )
        all_ok = False

    if all_ok:
        print(
            f"OK: {total} arquivos de "
            f"{n_genres} gêneros copiados com sucesso."
        )
    else:
        print("Verificação concluída com avisos.")


def cleanup():
    if GTZAN_RAW.exists():
        shutil.rmtree(GTZAN_RAW)
        print("Diretório temporário removido: data/gtzan_raw/")


def main():
    download()
    extract()
    counts = copy_files()
    verify(counts)
    cleanup()


if __name__ == "__main__":
    main()
