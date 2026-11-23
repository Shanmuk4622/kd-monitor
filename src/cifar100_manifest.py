"""Build unlabeled evaluation role manifests from the official CIFAR-100 binary archive."""

import csv
import hashlib
import json
import tarfile
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ARCHIVE_MD5 = "03b5dce01913d631647c71ecec9e9cb8"
RECORD_BYTES = 2 + 3 * 32 * 32


def split_rank(image_id):
    return hashlib.sha256(f"kd-view-v1|1729|{image_id}".encode("ascii")).digest()


def make_roles(train_labels, train_hashes, test_hashes, fit_per_class):
    if len(train_hashes) != len(train_labels):
        raise ValueError("Training labels and pixel digests differ in length")
    members = defaultdict(list)
    train_ids = [f"train/{i:05d}" for i in range(len(train_labels))]
    for i, label in enumerate(train_labels):
        members[int(label)].append(i)
    fit = set()
    for indices in members.values():
        if len(indices) <= fit_per_class:
            raise ValueError("Class lacks sufficient validation candidates")
        ordered = sorted(indices, key=lambda i: (split_rank(train_ids[i]), train_ids[i]))
        fit.update(ordered[:fit_per_class])
    seen_fit = {train_hashes[i] for i in fit}
    seen_val = set()
    rows = []
    for i, (image_id, label, pixel_sha) in enumerate(zip(train_ids, train_labels, train_hashes)):
        if i in fit:
            role, reason = "fit", ""
        elif pixel_sha in seen_fit:
            role, reason = "val_excluded", "matches_fit"
        elif pixel_sha in seen_val:
            role, reason = "val_excluded", "repeated_validation_pixels"
        else:
            role, reason = "validation", ""
            seen_val.add(pixel_sha)
        rows.append(dict(image_id=image_id, role=role, fine_label=int(label),
                         rgb_chw_sha256=pixel_sha, exclusion=reason))
    all_train = set(train_hashes)
    seen_test = set()
    for i, pixel_sha in enumerate(test_hashes):
        if pixel_sha in all_train:
            reason = "matches_official_training"
        elif pixel_sha in seen_test:
            reason = "repeated_test_pixels"
        else:
            reason = ""
        seen_test.add(pixel_sha)
        rows.append(dict(image_id=f"test/{i:05d}", role="test_excluded" if reason else "test_primary",
                         fine_label="", rgb_chw_sha256=pixel_sha, exclusion=reason))
    return rows


def check_split_fixture():
    labels = [0] * 4 + [1] * 4
    train_ids = [f"train/{i:05d}" for i in range(8)]
    by_rank = sorted(range(4), key=lambda i: split_rank(train_ids[i]))
    hashes = [str(i) for i in range(8)]
    hashes[by_rank[-1]] = hashes[by_rank[0]]
    rows = make_roles(labels, hashes, [hashes[by_rank[0]], "new", "new", "other"], 2)
    counts = Counter(r["role"] for r in rows)
    expected = {"fit": 4, "validation": 3, "val_excluded": 1, "test_primary": 2, "test_excluded": 2}
    if counts != expected or any(r["fine_label"] != "" for r in rows if r["image_id"].startswith("test/")):
        raise RuntimeError("Split fixture failed")
    return dict(counts)


def audit_cifar100(archive_path, output_parent):
    archive_path = Path(archive_path)
    md5, sha = hashlib.md5(), hashlib.sha256()
    with archive_path.open("rb") as file:
        for data in iter(lambda: file.read(1024 * 1024), b""):
            md5.update(data)
            sha.update(data)
    if md5.hexdigest() != ARCHIVE_MD5:
        raise ValueError("Archive differs from the upstream CIFAR-100 binary edition")
    with tarfile.open(archive_path, "r:gz") as tar:
        def read_split(name, expected):
            entry = tar.getmember(f"cifar-100-binary/{name}.bin")
            if not entry.isfile() or entry.size != expected * RECORD_BYTES:
                raise ValueError(f"Unexpected {name} binary data")
            with tar.extractfile(entry) as stream:
                raw = np.frombuffer(stream.read(), dtype=np.uint8).reshape(expected, RECORD_BYTES)
            labels = raw[:, 1].astype(int).tolist() if name == "train" else []
            hashes = [hashlib.sha256(row[2:]).hexdigest() for row in raw]
            return labels, hashes
        labels, train_hashes = read_split("train", 50000)
        _, test_hashes = read_split("test", 10000)
    if Counter(labels) != {i: 500 for i in range(100)}:
        raise ValueError("Unexpected fine-class support")
    rows = make_roles(labels, train_hashes, test_hashes, fit_per_class=450)
    counts = Counter(row["role"] for row in rows)
    if len(rows) != 60000 or counts["fit"] != 45000:
        raise RuntimeError("Split size mismatch")
    root = Path(output_parent)
    root.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="cifar100-manifest-", dir=root))
    csv_path = out / "split_manifest.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    result = dict(archive_md5=md5.hexdigest(), archive_sha256=sha.hexdigest(),
                  manifest_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                  counts=dict(counts), test_labels_exported=False)
    (out / "split_receipt.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return out, result
