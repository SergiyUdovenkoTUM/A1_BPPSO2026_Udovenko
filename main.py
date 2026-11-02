from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

XES_PATH = DATA_DIR / "BPIC17.xes"
CSV_PATH = DATA_DIR / "BPIC17.csv"


def main():
    for path in (XES_PATH, CSV_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found: {path}")

        print(f"Found dataset: {path.name}")

    print("Project setup is ready.")


if __name__ == "__main__":
    main()