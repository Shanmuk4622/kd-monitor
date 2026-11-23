"""CIFAR-100 training cohort, deterministic replay and quota-matched teacher banks."""

import csv
import tarfile
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from cache_preflight import allocate, key, pilot_scores, view_codes
from cifar100_manifest import RECORD_BYTES, audit_cifar100
from kd_fit_io import csv_write, file_info, hash_object, json_read, json_write, np_write


class Images:
    def __init__(self, rgb, labels, ids, fit, validation, classes, identity, padding):
        self.rgb = rgb
        self.labels = np.asarray(labels, dtype=np.int64)
        self.ids = list(ids)
        self.fit = np.asarray(fit, dtype=np.int64)
        self.validation = np.asarray(validation, dtype=np.int64)
        self.classes, self.identity, self.padding = classes, identity, padding
        total = np.zeros(3, dtype=np.float64)
        squared = total.copy()
        for start in range(0, len(self.fit), 256):
            a = self.rgb[self.fit[start : start + 256]].astype(np.float64) / 255
            total += a.sum(axis=(0, 2, 3))
            squared += (a * a).sum(axis=(0, 2, 3))
        n = len(self.fit) * rgb.shape[2] * rgb.shape[3]
        self.mean = total / n
        self.std = np.sqrt(squared / n - self.mean**2)
        if not np.all(self.std > 0):
            raise ValueError("A fitting channel has zero variance")

    def inputs(self, indices, codes, device):
        a = torch.from_numpy(self.rgb[indices]).to(device)
        if codes is not None:
            size, pad = a.shape[-1], self.padding
            flags = torch.as_tensor(codes, dtype=torch.long, device=device)
            crop, flip = flags // 2, flags % 2
            top, left = crop // (2 * pad + 1), crop % (2 * pad + 1)
            x = torch.arange(size, device=device)
            cols = torch.where(flip[:, None].bool(), size - 1 - x, x) + left[:, None]
            pos = (top[:, None, None] + x[None, :, None]) * (size + 2 * pad) + cols[:, None, :]
            padded = F.pad(a, (pad,) * 4, mode="reflect").flatten(2)
            a = padded.gather(2, pos.reshape(len(indices), 1, -1).expand(-1, 3, -1)).reshape_as(a)
        a = a.float().div_(255)
        mu = torch.as_tensor(self.mean, device=device, dtype=a.dtype)[None, :, None, None]
        sd = torch.as_tensor(self.std, device=device, dtype=a.dtype)[None, :, None, None]
        return (a - mu) / sd, torch.as_tensor(self.labels[indices], device=device)


def official_images(archive, root):
    """Source archive is hashed every launch. Teacher/student access only the official training member."""
    archive, root = Path(archive), Path(root)
    meta = file_info(archive)
    index_file = root / "INPUT.json"
    root.mkdir(parents=True, exist_ok=True)
    if index_file.exists():
        saved = json_read(index_file)
        path = root / saved["manifest"]
        if meta != saved["archive"] or file_info(path)["sha256"] != saved["split"]["manifest_sha256"]:
            raise RuntimeError("Original archive or frozen split differs")
    else:
        out, receipt = audit_cifar100(archive, root)
        path = out / "split_manifest.csv"
        saved = dict(archive=meta, split=receipt, manifest=path.relative_to(root).as_posix())
        json_write(index_file, saved)
    with tarfile.open(archive, "r:gz") as tar:
        with tar.extractfile("cifar-100-binary/train.bin") as stream:
            raw = np.frombuffer(stream.read(), dtype=np.uint8).reshape(50000, RECORD_BYTES)
    with path.open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["image_id"].startswith("train/")]
    ids = [r["image_id"] for r in rows]
    if len(rows) != len(raw) or ids != [f"train/{i:05d}" for i in range(50000)]:
        raise ValueError("Source order mismatch")
    if not np.array_equal(np.asarray([int(r["fine_label"]) for r in rows]), raw[:, 1]):
        raise ValueError("Source label mismatch")
    return Images(
        raw[:, 2:].reshape(-1, 3, 32, 32),
        raw[:, 1],
        ids,
        [i for i, r in enumerate(rows) if r["role"] == "fit"],
        [i for i, r in enumerate(rows) if r["role"] == "validation"],
        100,
        saved["split"]["manifest_sha256"],
        4,
    )


def toy_images():
    rng = np.random.default_rng(311)
    a = rng.integers(0, 256, (32, 3, 12, 12), dtype=np.uint8)
    y = np.arange(32) % 4
    from hashlib import sha256

    return Images(
        a,
        y,
        [f"toy/{i:02d}" for i in range(32)],
        np.arange(24),
        np.arange(24, 32),
        4,
        sha256(a.tobytes() + y.tobytes()).hexdigest(),
        2,
    )


def source_order(n, seed, epoch):
    return np.random.default_rng(key("source-order", seed, epoch)).permutation(n)


def fresh_codes(data, seed, epoch):
    rng = np.random.default_rng(key("fresh", seed, epoch))
    return rng.integers(0, 2 * (2 * data.padding + 1) ** 2, size=len(data.fit))


def build_bank(store, data, teacher, device, seed, policy, checkpoint_hook, group=4096):
    """Each phase is chunked. Progress is committed before the periodic publication hook."""
    state = store.load() or {"phase": "probes", "position": 0, "update": 0, "complete": False}
    ids = [data.ids[i] for i in data.fit]
    labels = data.labels[data.fit].tolist()
    codes = np.asarray([view_codes(i, seed, data.padding) for i in ids], dtype=np.int16)
    n = len(ids)
    teacher.eval().requires_grad_(False)

    def query_chunks(phase, positions, slots):
        for start in range(state["position"], len(positions), group):
            end = min(start + group, len(positions))
            pieces = []
            with torch.inference_mode():
                for first in range(start, end, 128):
                    last = min(first + 128, end)
                    ii, jj = positions[first:last], slots[first:last]
                    x, _ = data.inputs(data.fit[ii], codes[ii, jj], device)
                    pieces.append(teacher(x).cpu().numpy())
            name = f"cache/{phase}_{start:08d}.npy"
            out = np.concatenate(pieces).astype(np.float32, copy=False)
            if not np.isfinite(out).all():
                raise ValueError("Nonfinite cache logits")
            np_write(store.path(name), out)
            state.update(position=end, update=state["update"] + end - start)
            store.commit(state, [name])
            checkpoint_hook(store, False)

    def read_chunks(phase):
        names = sorted(k for k in store.pointer["files"] if k.startswith(f"cache/{phase}_"))
        return np.concatenate([np.load(store.path(name), allow_pickle=False) for name in names])

    if not state["complete"]:
        probe_i, probe_j = np.repeat(np.arange(n), 2), np.tile(np.arange(2), n)
        if state["phase"] == "probes":
            query_chunks("probe", probe_i, probe_j)
            state.update(phase="extra", position=0)
            store.commit(state, [])
        probes = read_chunks("probe").reshape(n, 2, data.classes)
        js, entropy = pilot_scores(torch.from_numpy(probes))
        quotas = allocate(ids, labels, entropy if policy == "entropy" else js, policy, seed)
        counts = np.asarray([quotas[i] for i in ids], dtype=np.int16)
        extra_i = np.repeat(np.arange(n), counts - 2)
        extra_j = np.concatenate([np.arange(2, c) for c in counts if c > 2])
        query_chunks("extra", extra_i, extra_j)
        extras = read_chunks("extra")
        offsets = np.cumsum(counts, dtype=np.int64) - counts
        packed = np.empty((4 * n, data.classes), dtype=np.float32)
        packed[offsets] = probes[:, 0]
        packed[offsets + 1] = probes[:, 1]
        packed[offsets[extra_i] + extra_j] = extras
        if counts.sum() != 4 * n or state["update"] != 4 * n:
            raise RuntimeError("Teacher calls and target row budget differ")
        np_write(store.path("bank/logits.npy"), packed)
        np_write(store.path("bank/counts.npy"), counts)
        np_write(store.path("bank/codes.npy"), codes)
        csv_write(
            store.path("bank/scores.csv"),
            ["image_id", "fine_label", "js_nats", "entropy_nats", "correct_probes", "view_count"],
            (
                (
                    ids[i],
                    labels[i],
                    js[i],
                    entropy[i],
                    int((probes[i].argmax(-1) == labels[i]).sum()),
                    counts[i],
                )
                for i in range(n)
            ),
        )
        csv_write(
            store.path("bank/views.csv"),
            ["row", "image_id", "slot", "crop_flip_code"],
            ((offsets[i] + s, ids[i], s, codes[i, s]) for i in range(n) for s in range(counts[i])),
        )
        state.update(phase="complete", position=0, complete=True)
        store.commit(
            state,
            ["bank/" + n for n in ("logits.npy", "counts.npy", "codes.npy", "scores.csv", "views.csv")],
        )
        checkpoint_hook(store, True)
    counts = np.load(store.path("bank/counts.npy"), allow_pickle=False)
    return dict(
        counts=counts,
        offsets=np.cumsum(counts, dtype=np.int64) - counts,
        codes=np.load(store.path("bank/codes.npy"), allow_pickle=False),
        logits=np.load(store.path("bank/logits.npy"), mmap_mode="r", allow_pickle=False),
        slot_offsets=np.array([key("slot-offset", seed, i) % int(c) for i, c in zip(ids, counts)]),
        identity=hash_object({"contract": store.tag, "logits": store.pointer["files"]["bank/logits.npy"]}),
    )
