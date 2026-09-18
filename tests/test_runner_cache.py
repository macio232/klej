from pathlib import Path

from klej.runner_cache import install_runner


def test_runner_is_content_addressed_and_standalone(tmp_path: Path) -> None:
    first = install_runner(tmp_path)
    second = install_runner(tmp_path)

    assert first == second
    assert first.parent.parent == tmp_path / ".klej" / "runners"
    assert "from klej" not in first.read_text()
    assert "import klej" not in first.read_text()
