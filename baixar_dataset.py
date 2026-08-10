import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "00_download.py",
    "01_extract.py",
    "02_convert.py",
    "03_select_segment.py",
    "04_build_dataset.py",
    "05_add_gtzan.py",
    "06_extract_features.py",
]

BASE = Path("data_loader/scripts")


def main():
    for script in SCRIPTS:
        path = BASE / script
        print(f"\n=== Executando {script} ===")
        ret = subprocess.run([sys.executable, str(path)])
        if ret.returncode != 0:
            print(f"ERRO: {script} falhou (código {ret.returncode})")
            sys.exit(1)

    print("\n=== Dataset pronto! ===")


if __name__ == "__main__":
    main()
