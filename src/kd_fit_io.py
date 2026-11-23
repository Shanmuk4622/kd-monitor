"""Durable local tensors, direct evidence files and a complete-update checkpoint."""

import csv
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def hash_object(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_info(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as reader:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            h.update(chunk)
    return {"bytes": Path(path).stat().st_size, "sha256": h.hexdigest()}


def safe_path(root, relative):
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Invalid relative artifact path")
    root = Path(root).resolve()
    p = (root / relative).resolve()
    if not p.is_relative_to(root) or p == root:
        raise ValueError("Artifact path outside its run directory")
    return p


def write_atomic(path, writer, binary=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".partial")
    mode = "wb" if binary else "w"
    options = {} if binary else {"encoding": "utf-8", "newline": ""}
    with part.open(mode, **options) as handle:
        writer(handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(part, path)
    if os.name == "posix":
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def json_write(path, value):
    write_atomic(path, lambda f: f.write(canonical(value)))


def json_read(path):
    return json.loads(Path(path).read_bytes())


def csv_write(path, fields, rows):
    def write(f):
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(fields)
        writer.writerows(rows)

    write_atomic(path, write, binary=False)


def np_write(path, array):
    write_atomic(path, lambda f: np.save(f, array, allow_pickle=False))


def torch_write(path, obj):
    write_atomic(path, lambda f: torch.save(obj, f))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def rng_restore(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise RuntimeError("Checkpoint GPU topology differs")
        torch.cuda.set_rng_state_all(state["cuda"])


class Checkpoints:
    """Only POINTER.json selects committed evidence. Unselected crash tails are ignored."""

    def __init__(self, root, identity):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self.tag = hash_object(identity)
        self.pointer_path = self.root / "POINTER.json"
        self.pointer = json_read(self.pointer_path) if self.pointer_path.exists() else None
        if self.pointer and self.pointer["identity"] != self.tag:
            raise RuntimeError("Checkpoint identity mismatch. Preserve the existing run.")

    def path(self, relative):
        return safe_path(self.root, relative)

    def load(self):
        if not self.pointer:
            return None
        for name, meta in self.pointer["files"].items():
            if file_info(self.path(name)) != meta:
                raise RuntimeError("Corrupt committed file: " + name)
        state = torch.load(self.path(self.pointer["state_file"]), map_location="cpu", weights_only=False)
        if state["identity"] != self.tag or state["generation"] != self.pointer["generation"]:
            raise RuntimeError("State and selecting pointer disagree")
        return state["payload"]

    def commit(self, payload, names, fault=None):
        previous = self.pointer or {}
        generation = previous.get("generation", 0) + 1
        files = dict(previous.get("files", {}))
        if previous:
            files.pop(previous["state_file"])
        for name in names:
            info = file_info(self.path(name))
            if name in files and files[name] != info:
                raise RuntimeError("Committed evidence would change: " + name)
            files[name] = info
        state_name = f"rolling/state_{generation % 2}.pt"
        torch_write(self.path(state_name), dict(identity=self.tag, generation=generation, payload=payload))
        files[state_name] = file_info(self.path(state_name))
        if fault == "before_pointer":
            raise InterruptedError("Injected power loss before pointer commit")
        row = {
            "generation": generation,
            "identity": self.tag,
            "update": payload.get("update", 0),
            "previous_chain": previous.get("chain"),
            "state_sha256": files[state_name]["sha256"],
            "new_files": {name: files[name] for name in names},
        }
        chain = hash_object(row)
        ledger_name = f"ledger/{generation:06d}.json"
        json_write(self.path(ledger_name), row)
        files[ledger_name] = file_info(self.path(ledger_name))
        pointer = dict(
            identity=self.tag,
            generation=generation,
            chain=chain,
            state_file=state_name,
            update=payload.get("update", 0),
            complete=payload.get("complete", False),
            files=files,
        )
        json_write(self.pointer_path, pointer)
        self.pointer = pointer
        return pointer

    def snapshot_files(self):
        if self.pointer is None:
            return {}
        return {**self.pointer["files"], "POINTER.json": file_info(self.pointer_path)}
