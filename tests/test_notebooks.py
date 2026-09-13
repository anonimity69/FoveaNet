from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def test_notebooks_are_valid_and_outputs_cleared():
    notebooks = sorted(ROOT.glob("foveanet_*.ipynb"))
    assert len(notebooks) == 6
    for path in notebooks:
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type == "code":
                assert cell.execution_count is None
                assert not cell.outputs
                # IPython magic is valid notebook syntax, not ordinary Python.
                if not cell.source.lstrip().startswith("%"):
                    compile(cell.source, str(path), "exec")
