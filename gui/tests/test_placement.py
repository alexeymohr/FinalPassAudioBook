from pathlib import Path

from fpab_gui import placement


def _csv(tmp_path: Path, name: str = "src.csv") -> Path:
    source = tmp_path / name
    source.write_text("data\n")
    return source


def test_places_stem_csv(tmp_path):
    target = placement.into_folder(_csv(tmp_path), tmp_path, "chapter", set())
    assert target == tmp_path / "chapter.csv"
    assert target.read_text() == "data\n"


def test_never_replaces_an_existing_file(tmp_path):
    existing = tmp_path / "chapter.csv"
    existing.write_text("old\n")
    target = placement.into_folder(_csv(tmp_path), tmp_path, "chapter", set())
    assert target == tmp_path / "chapter (2).csv"
    assert existing.read_text() == "old\n"


def test_respects_taken_in_run(tmp_path):
    taken: set[str] = set()
    first = placement.into_folder(_csv(tmp_path, "a.csv"), tmp_path, "chapter", taken)
    second = placement.into_folder(_csv(tmp_path, "b.csv"), tmp_path, "chapter", taken)
    assert first == tmp_path / "chapter.csv"
    assert second == tmp_path / "chapter (2).csv"


def test_skips_a_number_another_audio_file_owns(tmp_path):
    (tmp_path / "chapter.csv").write_text("old\n")
    (tmp_path / "chapter (2).wav").write_bytes(b"")
    target = placement.into_folder(_csv(tmp_path), tmp_path, "chapter", set())
    assert target == tmp_path / "chapter (3).csv"


def test_beside_or_numbered(tmp_path):
    wav = tmp_path / "book.wav"
    wav.write_bytes(b"")
    target = placement.beside_or_numbered(_csv(tmp_path), wav, set())
    assert target == tmp_path / "book.csv"


def test_places_without_o_nofollow(monkeypatch, tmp_path):
    monkeypatch.setattr(placement, "_NOFOLLOW", 0)   # as on Windows
    first = placement.into_folder(_csv(tmp_path), tmp_path, "c", set())
    second = placement.into_folder(_csv(tmp_path, "d.csv"), tmp_path, "c", set())
    assert first.name == "c.csv"
    assert second.name == "c (2).csv"

