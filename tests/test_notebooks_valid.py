"""
Static checks for the course-extension notebooks (05 and 06).

These do not execute the notebooks (that needs the model and dataset). They guard against
the cheap failures: invalid notebook JSON, code cells that do not parse, hard-coded absolute
paths, and notebooks that stop reusing the shared modules.
"""
import ast
import glob
import os

import pytest

nbformat = pytest.importorskip("nbformat")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NOTEBOOKS = sorted(glob.glob(os.path.join(ROOT, "notebooks", "0[56]_*.ipynb")))


def _code_cells(path):
    nb = nbformat.read(path, as_version=4)
    return nb, [c for c in nb.cells if c.cell_type == "code"]


def test_both_course_notebooks_exist():
    names = [os.path.basename(p)[:2] for p in NOTEBOOKS]
    assert names == ["05", "06"], f"expected notebooks 05 and 06, found {names}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: os.path.basename(p))
def test_notebook_is_valid_nbformat(path):
    nb, _ = _code_cells(path)
    nbformat.validate(nb)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: os.path.basename(p))
def test_every_code_cell_parses(path):
    _, cells = _code_cells(path)
    assert cells, "notebook has no code cells"
    for i, cell in enumerate(cells, 1):
        try:
            ast.parse(cell.source)
        except SyntaxError as exc:
            pytest.fail(f"{os.path.basename(path)} code cell {i} has a syntax error: {exc}")


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: os.path.basename(p))
def test_cells_print_their_results(path):
    _, cells = _code_cells(path)
    with_output = [c for c in cells if "print(" in c.source or "plt.show()" in c.source]
    assert len(with_output) / len(cells) >= 0.8, "most cells should print or plot what they did"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: os.path.basename(p))
def test_no_hardcoded_absolute_paths_and_shared_modules_reused(path):
    _, cells = _code_cells(path)
    code = "\n".join(c.source for c in cells)
    for bad in ["C:\\", "C:/", "/Users/", "/home/", "OneDrive"]:
        assert bad not in code, f"hard-coded path fragment {bad!r} in {os.path.basename(path)}"
    assert "import data_pipeline" in code and "import inference_utils" in code, \
        "notebooks must import the existing/shared modules instead of copying their logic"
