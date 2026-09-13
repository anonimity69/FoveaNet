"""Export the review files into a new repository without local data or history."""

import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL = {
    ".gitignore",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    "FOVEANET_LOG.txt",
    "FOVEANET_PROGRESS.txt",
    "PREDICTIONS.txt",
}
DIRECTORIES = ("docs", "tests", "tools", ".github", "figures", "report/figures")
EXCLUDED = {"__pycache__", ".DS_Store", ".ipynb_checkpoints", ".pytest_cache", ".ruff_cache"}


def review_files(root=ROOT):
    files = {root / name for name in TOP_LEVEL}
    files.update(root.glob("*.py"))
    files.update(root.glob("foveanet_*.ipynb"))
    for directory in DIRECTORIES:
        files.update((root / directory).rglob("*"))
    return sorted(
        p
        for p in files
        if p.is_file()
        and not p.is_symlink()
        and not any(part in EXCLUDED or part.startswith("._") for part in p.relative_to(root).parts)
    )


def export(destination):
    destination = destination.resolve()
    if destination == ROOT or ROOT in destination.parents:
        raise ValueError("Choose a destination outside the source repository")
    destination.mkdir(parents=True, exist_ok=False)
    for path in review_files():
        target = destination / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    subprocess.run(["git", "init", "--initial-branch=main", str(destination)], check=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="new folder outside this repository")
    args = parser.parse_args()
    print(export(args.destination))


if __name__ == "__main__":
    main()
