from __future__ import annotations

import sys
from pathlib import Path

import pytest


PREPROCESSING = Path(__file__).resolve().parents[1] / "preprocessing"
sys.path.insert(0, str(PREPROCESSING))

from prepare_dstc11 import _raw_file as dstc11_raw_file  # noqa: E402
from prepare_dstc9 import _raw_file as dstc9_raw_file  # noqa: E402
from prepare_ketod import _single_path as ketod_single_path  # noqa: E402


def make_raw_file(root: Path, repository: str, split: str, name: str = "logs.json") -> Path:
    path = root / repository / "data" / split / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


@pytest.mark.parametrize(
    ("repository", "locator"),
    [
        ("dstc9", dstc9_raw_file),
        ("alexa-with-dstc9-track1-dataset", dstc9_raw_file),
        ("dstc11", dstc11_raw_file),
        ("dstc11-track5", dstc11_raw_file),
    ],
)
def test_alias_and_official_clone_layouts(tmp_path: Path, repository: str, locator) -> None:
    expected = make_raw_file(tmp_path, repository, "train")
    assert locator(tmp_path, "train", "logs.json") == expected


@pytest.mark.parametrize(
    ("repositories", "locator", "benchmark"),
    [
        (("dstc9", "alexa-with-dstc9-track1-dataset"), dstc9_raw_file, "DSTC9"),
        (("dstc11", "dstc11-track5"), dstc11_raw_file, "DSTC11"),
    ],
)
def test_multiple_candidates_are_ambiguous(
    tmp_path: Path, repositories: tuple[str, str], locator, benchmark: str
) -> None:
    for repository in repositories:
        make_raw_file(tmp_path, repository, "val", "labels.json")
    with pytest.raises(RuntimeError, match=rf"ambiguous {benchmark} data/val/labels.json"):
        locator(tmp_path, "val", "labels.json")


@pytest.mark.parametrize(
    ("locator", "benchmark"),
    [(dstc9_raw_file, "DSTC9"), (dstc11_raw_file, "DSTC11")],
)
def test_missing_file_has_clear_error(tmp_path: Path, locator, benchmark: str) -> None:
    with pytest.raises(FileNotFoundError, match=rf"{benchmark} raw file not found"):
        locator(tmp_path, "train", "logs.json")


def test_benchmark_token_must_be_a_component_token(tmp_path: Path) -> None:
    make_raw_file(tmp_path, "notdstc9benchmark", "train")
    with pytest.raises(FileNotFoundError, match="DSTC9 raw file not found"):
        dstc9_raw_file(tmp_path, "train", "logs.json")


def test_ketod_compressed_release_has_extraction_hint(tmp_path: Path) -> None:
    archive = tmp_path / "ketod" / "ketod_release.zip"
    archive.parent.mkdir()
    archive.touch()
    with pytest.raises(FileNotFoundError, match="appears to still be compressed") as error:
        ketod_single_path(tmp_path, "train_ketod.json")
    assert "train_ketod.json and test_ketod.json" in str(error.value)
