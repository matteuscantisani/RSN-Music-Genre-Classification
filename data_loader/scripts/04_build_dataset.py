from pathlib import Path

SEG_DIR = Path("data_loader/data/segments")
DST_DIR = Path("data_loader/data/final_dataset")


def build_dataset():
    DST_DIR.mkdir(parents=True, exist_ok=True)

    genre_dirs = sorted(d for d in SEG_DIR.iterdir() if d.is_dir())
    if not genre_dirs:
        raise FileNotFoundError(
            "Nenhum diretório de gênero encontrado em data/segments/. "
            "Execute 03_select_segment.py primeiro."
        )

    for genre_dir in genre_dirs:
        genre = genre_dir.name
        wav_files = sorted(f for f in genre_dir.iterdir() if f.suffix == ".wav")
        if not wav_files:
            continue

        for i, wav_path in enumerate(wav_files):
            dst_name = f"{genre}.{i:05d}.wav"
            dst_path = DST_DIR / dst_name
            wav_path.rename(dst_path)
            print(f"Movido: {wav_path.name} -> {dst_name}")


if __name__ == "__main__":
    build_dataset()
