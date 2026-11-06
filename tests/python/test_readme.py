"""Every ```python block in README.md and docs/*.md must run."""

import os
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
BLOCK = re.compile(r"```python\n(.*?)```", re.S)


def blocks():
    for path in DOCS:
        for i, code in enumerate(BLOCK.findall(path.read_text())):
            if "# doctest: skip" not in code:
                yield pytest.param(code, id=f"{path.name}-{i}")


@pytest.mark.parametrize("code", list(blocks()))
def test_doc_example_runs(code, tmp_path):
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        exec(compile(code, "<doc>", "exec"), {"__name__": "__doc_example__"})
    finally:
        os.chdir(cwd)
