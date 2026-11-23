"""Resolve the Tiny ImageNet mount before entering the published experiment runtime."""

from pathlib import Path


def tiny_input_path(source, override=None, base=Path("/kaggle/input")):
    owner, slug = source.split("/")
    if not owner or not slug or owner in (".", "..") or slug in (".", ".."):
        raise ValueError("Set TINY_KAGGLE_SOURCE to the attached Kaggle dataset's owner/slug.")
    base = base.resolve()
    if override:
        root = Path(override).resolve()
        if not root.is_dir() or not root.is_relative_to(base) or slug not in root.parts:
            raise ValueError(f"Set INPUT_PATHS['tiny_imagenet200'] to the mounted folder for {source}.")
    else:
        owned = base / "datasets" / owner / slug
        roots = [p for p in (owned, base / slug) if p.is_dir()]
        if not roots:
            roots = [p for p in base.rglob(slug) if p.is_dir()]
        if len(roots) != 1:
            raise ValueError(f"Attach {source} or specify its exact folder in INPUT_PATHS. Found {roots}.")
        root = roots[0]

    # Check the canonical layout before walking a mount with 120,000 images.
    for folder in (root / "tiny-imagenet-200", root):
        if ((folder / "wnids.txt").is_file() and (folder / "train").is_dir()
                and (folder / "val").is_dir() and not (folder / "tiny-imagenet-200").is_dir()):
            return str(folder)
    markers = [p for p in root.rglob("wnids.txt") if p.is_file() and "__MACOSX" not in p.parts]
    if not markers:
        # The experiment reader also supports the original, unopened Stanford ZIP.
        return str(root)

    candidates = []
    for marker in markers:
        folder = marker.parent
        # The reader must see a single class list beneath the selected folder.
        if any(other != marker and other.is_relative_to(folder) for other in markers):
            continue
        if (folder / "train").is_dir() and (folder / "val").is_dir():
            candidates.append(folder)
    if len(candidates) != 1:
        raise ValueError(
            f"Found {len(candidates)} self-contained Tiny ImageNet folders in {root}: {candidates}. "
            "Select one complete folder in INPUT_PATHS['tiny_imagenet200'], or attach the original Stanford ZIP."
        )
    return str(candidates[0])
