import shutil
from pathlib import Path
from pydub import AudioSegment

RAW_DIR = Path("data_loader/data/raw")
WAV_DIR = Path("data_loader/data/wav")

_ffmpeg = shutil.which("ffmpeg") or (Path.home() / ".local/bin/ffmpeg")
if _ffmpeg:
    AudioSegment.converter = str(_ffmpeg)
_ffprobe = shutil.which("ffprobe") or (Path.home() / ".local/bin/ffprobe")
if _ffprobe:
    AudioSegment.ffprobe = str(_ffprobe)

def convert_all():
    genre_dirs = [d for d in RAW_DIR.iterdir() if d.is_dir()]
    if not genre_dirs:
        raise FileNotFoundError(
            "Nenhum diretório de gênero encontrado em data/raw/. "
            "Execute 01_extract.py primeiro para extrair os arquivos."
        )

    for genre_dir in sorted(genre_dirs):
        genre = genre_dir.name
        out_dir = WAV_DIR / genre
        out_dir.mkdir(parents=True, exist_ok=True)

        for audio_path in sorted(genre_dir.iterdir()):
            if not audio_path.is_file():
                continue
            stem = audio_path.stem
            out_path = out_dir / f"{stem}.wav"

            if out_path.exists():
                print(f"Já existe: {out_path}")
                continue

            audio = AudioSegment.from_file(audio_path)
            audio.export(out_path, format="wav")
            print(f"Convertido: {audio_path.name} -> {out_path}")

if __name__ == "__main__":
    convert_all()