import argparse
import tarfile
from pathlib import Path

from assets import load_catalog, load_deck

ROOT = Path(__file__).resolve().parent
FILES = ("main.py", "policy.py", "schema.py", "assets.py", "cards.json", "deck.csv")


def package(output: Path) -> None:
    cards, _ = load_catalog(ROOT)
    load_deck(ROOT, cards)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for name in FILES:
            archive.add(ROOT / name, arcname=name)
    if output.stat().st_size > 197.7 * 1024 * 1024:
        raise ValueError("Archive exceeds the published size limit")
    print(f"Public simulator candidate: {output} ({output.stat().st_size:,} bytes)")
    print("Current Playground entrypoint, CSV schema, and card eligibility require SDK validation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "dist/public-baseline.tar.gz")
    args = parser.parse_args()
    package(args.output)
