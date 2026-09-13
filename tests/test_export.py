from pathlib import Path

from tools.export_review import ROOT, review_files


def test_export_includes_review_files_but_not_local_state():
    paths = {p.relative_to(ROOT) for p in review_files()}
    assert Path("README.md") in paths
    assert Path("tools/export_review.py") in paths
    assert Path("requirements.txt") in paths
    assert Path("PREDICTIONS.txt") in paths
    for path in paths:
        assert path.parts[0] not in {"data", "node_modules"}
        assert not path.parts[0].startswith(".") or path.parts[0] in {".github", ".gitignore"}
        assert "__pycache__" not in path.parts
