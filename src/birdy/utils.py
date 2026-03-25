"""Basic utils function for the Birdy project."""

from pathlib import Path


def parse_path(path: Path | str) -> Path:
    """Parse the provided path and return the pathlib.Path object."""
    return Path(path)


def ensure_path_exist(*path: Path | str) -> None:
    """Ensure path exists by creating intermediate directories."""
    for p in path:
        parse_path(p).parent.mkdir(exist_ok=True, parents=True)


def ensure_dir_exist(*path: Path | str) -> None:
    """Ensure directories exists by creating intermediate directories."""
    for p in path:
        parse_path(p).mkdir(exist_ok=True, parents=True)
