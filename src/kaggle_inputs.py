"""Read the three permitted Kaggle mounts, including extracted or archived distributions."""

import copy
import hashlib
import io
import os
import pickle
import tarfile
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from cifar100_manifest import split_rank
from kd_fit_data import Images
from kd_fit_io import csv_write, hash_object, json_write

SOURCES = {
    "cifar10": "shanmuk4622/dataset-cifar10-python",
    "cifar100": "shanmuk4622/dataset-cifar100-python",
    "tiny_imagenet200": "shanmuk4622/tinyimagenet200",
}
SPECS = {"cifar10": (10, 32, 4500), "cifar100": (100, 32, 450), "tiny_imagenet200": (200, 64, 450)}
CIFAR_FILES = {
    "cifar10": {
        "data_batch_1": "c99cafc152244af753f735de768cd75f",
        "data_batch_2": "d4bba439e000b95fd0a9bffe97cbabec",
        "data_batch_3": "54ebc095f3ab1f0389bbae665268c751",
        "data_batch_4": "634d18415352ddfa80567beed471001a",
        "data_batch_5": "482c414d41f54cd18b22e5b47cb7c3cb",
        "test_batch": "40351d587109b95175f43aff81a1287e",
        "batches.meta": "5ff9c542aee3614f3951f8cda6e48888",
    },
    "cifar100": {
        "train": "16019d7e3df5f24257cddd939b257f8d",
        "test": "f0ef6b0ae62326f3e7ffdfab6717acfc",
        "meta": "7973b15100ade9c7d40fb424638fde48",
    },
}


def mount(dataset, override=None, base=Path("/kaggle/input")):
    owner, slug = SOURCES[dataset].split("/")
    if override:
        matches = [Path(override)]
    else:
        matches = [p for p in (base / "datasets" / owner / slug, base / slug) if p.is_dir()]
        if not matches:
            matches = [p for p in base.rglob(slug) if p.is_dir()]
    if (
        len(matches) != 1
        or not matches[0].is_dir()
        or not matches[0].resolve().is_relative_to(base.resolve())
        or slug not in matches[0].resolve().parts
    ):
        raise ValueError(
            f"Attach {SOURCES[dataset]} and specify its exact /kaggle/input folder in INPUT_PATHS. Candidates: {matches}"
        )
    return matches[0]


class Members:
    """A logical tree. Read archive members directly and never extract untrusted paths."""

    def __init__(self, root, stack, marker):
        self.root = Path(root)
        self.extracted = True
        self.names = []
        for directory, _, files in os.walk(self.root, onerror=self._raise_walk_error):
            prefix = Path(directory).relative_to(self.root)
            self.names.extend((prefix / f).as_posix() for f in files)
        self.read = lambda n: (self.root / n).read_bytes()
        if any(PurePosixPath(n).name == marker for n in self.names):
            return
        archives = [self.root / n for n in self.names if n.lower().endswith((".tar.gz", ".tgz", ".zip"))]
        if len(archives) != 1:
            raise ValueError(f"Expected {marker} or one dataset archive in {root}. Sample: {self.names[:12]}")
        archive = archives[0]
        self.extracted = False
        if zipfile.is_zipfile(archive):
            z = stack.enter_context(zipfile.ZipFile(archive))
            self.names = [n for n in z.namelist() if not n.endswith("/")]
            self.read = z.read
            nested = [n for n in self.names if n.endswith((".tar.gz", ".tgz"))]
            if not any(PurePosixPath(n).name == marker for n in self.names) and len(nested) == 1:
                tar = stack.enter_context(tarfile.open(fileobj=io.BytesIO(z.read(nested[0])), mode="r:gz"))
            else:
                return
        else:
            tar = stack.enter_context(tarfile.open(archive, "r:*"))
        self.names = [m.name for m in tar.getmembers() if m.isfile()]
        self.read = lambda n: tar.extractfile(n).read()

    @staticmethod
    def _raise_walk_error(error):
        raise error

    def by_name(self, name):
        hits = [n for n in self.names if PurePosixPath(n).name == name and "__MACOSX" not in n]
        if len(hits) != 1:
            raise ValueError(f"Expected one {name}, found {hits[:5]}")
        return hits[0]


def read_cifar(dataset, tree, release):
    arrays, labels, coarse, inventory = [], [], [], []
    small = dataset == "cifar10"
    for name, expected in CIFAR_FILES[dataset].items():
        path = tree.by_name(name)
        blob = tree.read(path)
        if hashlib.md5(blob).hexdigest() != expected:
            raise ValueError(
                f"{name} differs from the canonical Python batch; verify the attached Kaggle edition."
            )
        inventory.append((path, len(blob), hashlib.sha256(blob).hexdigest()))
        batch = pickle.loads(blob, encoding="latin1")
        if name in ("meta", "batches.meta"):
            names = list(batch["label_names" if small else "fine_label_names"])
            continue
        is_test = name in ("test_batch", "test")
        a = batch["data"]
        if a.dtype != np.uint8 or a.ndim != 2 or a.shape[1] != 3072:
            raise ValueError(f"Invalid RGB array in {name}")
        arrays.append(a.reshape(-1, 3, 32, 32))
        if not is_test or release:
            labels.extend(batch["labels" if small else "fine_labels"])
            if not small:
                coarse.extend(batch["coarse_labels"])
    return np.concatenate(arrays), labels, coarse, names, inventory


def read_tiny(tree, release, expected=(100000, 10000, 200)):
    wn = tree.by_name("wnids.txt")
    prefix = wn[: -len("wnids.txt")]
    names = tree.read(wn).decode().split()
    lookup = {n: i for i, n in enumerate(names)}
    members = sorted(
        n for n in tree.names if n.startswith(prefix) and n.lower().endswith((".jpeg", ".jpg", ".png"))
    )
    groups = [[n for n in members if n.startswith(prefix + part + "/")] for part in ("train", "val")]
    photos = groups[0] + groups[1]
    if (len(groups[0]), len(groups[1]), len(lookup)) != expected:
        raise ValueError(
            f"Expected {expected} (train images, val images, unique wnids), "
            f"found {(len(groups[0]), len(groups[1]), len(lookup))} under {prefix or tree.root}."
        )
    annotation = prefix + "val/val_annotations.txt"
    if annotation not in tree.names and any(
        not any(part in lookup for part in PurePosixPath(n).parts) for n in groups[1]
    ):
        raise ValueError("Missing val/val_annotations.txt for the flat Tiny ImageNet validation images.")
    metadata = [wn] + ([annotation] if annotation in tree.names else [])
    inventory = []
    for path in metadata:
        b = tree.read(path)
        inventory.append((path, len(b), hashlib.sha256(b).hexdigest()))
    valmap = {}
    if release and annotation in tree.names:
        valmap = {
            r.split()[0]: r.split()[1] for r in tree.read(annotation).decode().splitlines() if r.strip()
        }
    images = np.empty((len(photos), 3, 64, 64), np.uint8)
    labels = []
    ids = []
    def decode(path):
        b = tree.read(path)
        with Image.open(io.BytesIO(b)) as im:
            if im.size != (64, 64):
                raise ValueError(f"Unexpected size in {path}")
            pixels = np.asarray(im.convert("RGB")).transpose(2, 0, 1)
        return pixels, (path, len(b), hashlib.sha256(b).hexdigest())

    started = last_report = time.monotonic()
    print(f"Tiny ImageNet: decoding and hashing {len(photos):,} images from {tree.root}", flush=True)
    with ExitStack() as stack:
        pool = stack.enter_context(ThreadPoolExecutor(max_workers=8)) if tree.extracted else None
        for start in range(0, len(photos), 128):
            batch = photos[start:start + 128]
            records = pool.map(decode, batch) if pool else map(decode, batch)
            for i, (pixels, entry) in enumerate(records, start):
                images[i] = pixels
                inventory.append(entry)
            done = start + len(batch)
            now = time.monotonic()
            if now - last_report >= 30 or done == len(photos):
                print(f"Tiny ImageNet: {done:,}/{len(photos):,} images, {done/max(now-started, 0.001):.1f} images/s", flush=True)
                last_report = now
    for i, path in enumerate(photos):
        rel = path[len(prefix) :]
        ids.append(rel)
        if i < len(groups[0]) or release:
            wnid = (
                valmap.get(
                    PurePosixPath(path).name, next((part for part in rel.split("/") if part in lookup), "")
                )
                if i >= len(groups[0])
                else rel.split("/")[1]
            )
            if wnid not in lookup:
                raise ValueError(f"Missing Tiny ImageNet synset for {rel}")
            labels.append(lookup[wnid])
    return images, labels, [], names, inventory, ids


def load_dataset(dataset, root, out, version, release=False):
    classes, size, quota = SPECS[dataset]
    parts = Path(root).parts
    if version is None:
        version = next(
            (parts[i + 1] for i, p in enumerate(parts[:-1]) if p == "versions" and parts[i + 1].isdigit()),
            None,
        )
    out = Path(out)
    print(f"{dataset}: listing files in {root}", flush=True)
    with ExitStack() as stack:
        tree = Members(
            root,
            stack,
            {"cifar10": "data_batch_1", "cifar100": "train", "tiny_imagenet200": "wnids.txt"}[dataset],
        )
        if dataset == "tiny_imagenet200":
            rgb, y, coarse, names, inventory, ids = read_tiny(tree, release)
        else:
            rgb, y, coarse, names, inventory = read_cifar(dataset, tree, release)
            ids = [f"train/{i:05d}" for i in range(50000)] + [f"test/{i:05d}" for i in range(10000)]
    n = 100000 if classes == 200 else 50000
    if (
        rgb.shape != (n + 10000, 3, size, size)
        or len(set(names)) != classes
        or len(y) != (n + 10000 if release else n)
    ):
        raise ValueError(f"Unexpected dataset inventory for {dataset}")
    members = defaultdict(list)
    for i, c in enumerate(y[:n]):
        members[int(c)].append(i)
    if set(members) != set(range(classes)) or any(len(g) != n // classes for g in members.values()):
        raise ValueError("Invalid training class membership")
    fit = sorted(
        i
        for group in members.values()
        for i in sorted(group, key=lambda j: (split_rank(ids[j]), ids[j]))[:quota]
    )
    print(f"{dataset}: hashing decoded pixels and checking split overlaps", flush=True)
    hashes = [hashlib.sha256(a.tobytes()).hexdigest() for a in rgb]
    frozen = set(fit)
    seen_fit = {hashes[i] for i in fit}
    seen_train = set(hashes[:n])
    seen_val, seen_test = set(), set()
    val, clean, rows = [], [], []
    for i, (key, h) in enumerate(zip(ids, hashes)):
        why = ""
        if i in frozen:
            role = "fit"
        elif i < n:
            why = "matches_fit" if h in seen_fit else ("repeated_validation_pixels" if h in seen_val else "")
            role = "val_excluded" if why else "validation"
            if not why:
                val.append(i)
            seen_val.add(h)
        else:
            why = (
                "matches_official_training"
                if h in seen_train
                else ("repeated_test_pixels" if h in seen_test else "")
            )
            role = "test_excluded" if why else "test_primary"
            if not why:
                clean.append(i - n)
            seen_test.add(h)
        rows.append((key, role, y[i] if i < n else "", coarse[i] if coarse and i < n else "", h, why))
    identity = hash_object([SOURCES[dataset], names, inventory, rows])
    csv_write(
        out / "splits.csv", ["id", "role", "fine_label", "coarse_label", "pixel_sha256", "exclusion"], rows
    )
    csv_write(out / "source_files.csv", ["path", "bytes", "sha256"], inventory)
    print(f"{dataset}: computing normalization on {len(fit):,} fitting images", flush=True)
    data = Images(rgb[:n], y[:n], ids[:n], fit, val, classes, identity, 4 if size == 32 else 8)
    data.prefix = "kaggle-v2." + dataset
    json_write(
        out / "input.json",
        dict(
            source=SOURCES[dataset],
            version=version,
            version_pin="SHA-256 inventory and split",
            identity=identity,
            classes=names,
            mean=data.mean.tolist(),
            std=data.std.tolist(),
            fit=len(fit),
            validation=len(val),
            clean_final=len(clean),
        ),
    )
    test = None
    if release:
        if set(y[n:]) != set(range(classes)) or len(set(Counter(y[n:]).values())) != 1:
            raise ValueError("Unexpected final class counts")
        test = copy.copy(data)
        test.rgb, test.labels, test.ids = rgb[n:], np.array(y[n:], dtype=np.int64), ids[n:]
    return data, test, np.array(clean, dtype=np.int64)
