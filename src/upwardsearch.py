def _start_dirs() -> list[Path]:
    """Places to begin an upward search.

    `__file__` is missing (or misleading) inside Jupyter, so the current working
    directory has to be a first-class starting point — otherwise every path in
    this module silently resolves somewhere else when run from a notebook.
    """
    dirs = [Path.cwd().resolve()]
    try:
        dirs.append(Path(__file__).resolve().parent)
    except NameError:  # notebook / REPL
        pass
    return dirs


def _walk_up(start: Path):
    return [start, *start.parents]


def find_repo_root() -> Path:
    for start in _start_dirs():
        for directory in _walk_up(start):
            if (directory / "CLAUDE.md").is_file():
                return directory
    return Path.cwd().resolve()


REPO_ROOT = find_repo_root()
