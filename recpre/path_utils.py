import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_within_root(
    path: str | os.PathLike[str],
    root: str | os.PathLike[str],
    *,
    label: str,
) -> Path:
    """Resolve a path and reject absolute, relative, or symlink escapes from root."""
    resolved_root = Path(root).expanduser().resolve(strict=False)
    candidate = Path(os.path.expandvars(os.fspath(path))).expanduser()
    if not candidate.is_absolute():
        candidate = resolved_root / candidate
    resolved_candidate = candidate.resolve(strict=False)
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            f"{label} must stay inside project root {resolved_root}, got {resolved_candidate}"
        ) from exc
    return resolved_candidate


def resolve_project_path(path: str | os.PathLike[str], *, label: str) -> Path:
    return resolve_within_root(path, PROJECT_ROOT, label=label)
