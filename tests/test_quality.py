"""Image quality filter tests using synthetic PNG files."""

from pathlib import Path

from PIL import Image

from src.data.quality import ImageQualityFilter


def _write_png(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    Image.new("RGB", size, color).save(path)


def test_drops_corrupt_tiny_and_duplicates(tmp_path: Path):
    ok = tmp_path / "ok.png"
    tiny = tmp_path / "tiny.png"
    dup = tmp_path / "dup.png"
    bad = tmp_path / "bad.png"
    _write_png(ok, (64, 64), (10, 10, 10))
    _write_png(tiny, (8, 8), (10, 10, 10))
    _write_png(dup, (64, 64), (10, 10, 10))
    bad.write_bytes(b"not-an-image")

    report = ImageQualityFilter(min_width=32, min_height=32).filter(
        [(ok, 0), (tiny, 0), (dup, 1), (bad, 2)]
    )
    assert report.dropped_size == 1
    assert report.dropped_corrupt == 1
    assert report.dropped_duplicate == 1
    assert len(report.kept) == 1
    assert report.kept[0][0] == ok
