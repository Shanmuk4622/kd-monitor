"""Disposable fits and independent fresh-process recovery checks for the primary runner."""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from cache_preflight import render_view
from kd_fit_data import build_bank, toy_images
from kd_fit_engine import model_for, train_fit
from kd_fit_io import Checkpoints, file_info, json_read, json_write


def toy_config(arm="teacher"):
    return dict(seed=42, arm=arm, epochs=3, batch=8, warm_epochs=1, commit_updates=2, model="fixture")


def unit(path, arm, device):
    torch.set_num_threads(1)
    data = toy_images()
    cfg = toy_config(arm)
    store = Checkpoints(path, {"kind": "fixture", "cfg": cfg, "data": data.identity, "device": str(device)})
    return data, cfg, store


def resumed_worker(root, limit, fault, device):
    data, cfg, store = unit(root, "teacher", device)
    return train_fit(
        store, data, cfg, torch.device(device), lambda *_: None, until=limit, fault_update=fault
    )


def child(root, device, until=None, fault=None):
    args = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "--worker",
        str(root),
        "--device",
        str(device),
    ]
    if until is not None:
        args += ["--until", str(until)]
    if fault is not None:
        args += ["--fault", str(fault)]
    env = dict(os.environ)
    env.pop("HF_TOKEN", None)
    result = subprocess.run(
        args, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=240
    )
    if fault is None and result.returncode != 0:
        raise RuntimeError(result.stdout[-4000:])
    if fault is not None and (result.returncode == 0 or "Injected power loss" not in result.stdout):
        raise RuntimeError("Expected pre-pointer fault was not observed: " + result.stdout[-700:])


def same(left, right):
    if torch.is_tensor(left):
        return torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same(v, right[k]) for k, v in left.items())
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right


def predictions(store):
    actual = {}
    for name in sorted(store.pointer["files"]):
        if not name.endswith("/exposure.csv"):
            continue
        logits = np.load(store.path(name).with_name("student_logits.npy"), allow_pickle=False)
        with store.path(name).open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                k = (int(row["update"]), row["image_id"])
                if k in actual:
                    raise RuntimeError("Repeated committed training observation")
                actual[k] = logits[int(row["logit_row"])]
    return actual


def verify_continuation(reference, restored, device):
    _, _, a = unit(reference, "teacher", device)
    _, _, b = unit(restored, "teacher", device)
    s1, s2 = a.load(), b.load()
    for field in ("model", "optimizer", "rng", "partial", "update", "scheduler_update", "scaler", "best"):
        if not same(s1[field], s2[field]):
            raise RuntimeError("Restart diverged on " + field)
    obs_a, obs_b = predictions(a), predictions(b)
    if not same(obs_a, obs_b) or len(obs_a) != 72:
        raise RuntimeError("Restart changed the committed augmented predictions")
    return dict(
        updates=s1["update"],
        observations=len(obs_a),
        exact_model_optimizer_rng=True,
        exact_training_predictions=True,
    )


def qualification(root, device="cpu", all_arms=True):
    device = torch.device(device)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    baseline, recovered = root / "uninterrupted", root / "interrupted"
    child(baseline, device)
    child(recovered, device, until=2)
    child(recovered, device, fault=3)
    if json_read(recovered / "POINTER.json")["update"] != 2:
        raise RuntimeError("Crash advanced the selecting pointer")
    child(recovered, device)
    receipt = verify_continuation(baseline, recovered, device)
    data, _, control = unit(baseline, "teacher", device)
    teacher = model_for("fixture", data.classes).to(device).eval()
    endpoint = json_read(control.path("ENDPOINT.json"))
    teacher.load_state_dict(
        torch.load(control.path(endpoint["best"]["weights"]), map_location=device, weights_only=True)
    )
    xi = np.asarray([0, 3, 8])
    vi = np.asarray([0, 5, 18])
    actual, _ = data.inputs(xi, vi, device)
    ref = torch.stack(
        [render_view(torch.from_numpy(data.rgb[i].copy()), int(v), data.padding) for i, v in zip(xi, vi)]
    )
    expected = (
        ref.float().to(device) / 255
        - torch.tensor(data.mean, dtype=torch.float32, device=device)[None, :, None, None]
    )
    expected /= torch.tensor(data.std, dtype=torch.float32, device=device)[None, :, None, None]
    if not torch.equal(actual, expected):
        raise RuntimeError("Batched crop differs from independent crop replay")
    receipt["vectorized_crop_exact"] = True
    receipt["arms"] = {}
    if all_arms:
        for arm in ("hard_labels", "live_kd", "uniform", "shuffled", "entropy", "disagreement"):
            _, candidate, store = unit(root / "arms" / arm, arm, device)
            bank = None
            if arm not in ("hard_labels", "live_kd"):
                cstore = Checkpoints(
                    root / "banks" / arm, {"fixture": arm, "seed": 42, "data": data.identity}
                )
                bank = build_bank(cstore, data, teacher, device, 42, arm, lambda *_: None, group=16)
                if cstore.pointer["update"] != 96 or bank["logits"].shape != (96, 4):
                    raise RuntimeError("Incorrect cache call count or shape")
                # Actual selected teacher and stored target must agree on corresponding view tensors.
                positions = np.arange(6)
                slots = bank["counts"][positions] - 1
                samples, _ = data.inputs(data.fit[positions], bank["codes"][positions, slots], device)
                with torch.no_grad():
                    error = float(
                        np.max(
                            np.abs(
                                teacher(samples).cpu().numpy()
                                - bank["logits"][bank["offsets"][positions] + slots]
                            )
                        )
                    )
                if error > 1e-5:
                    raise RuntimeError("Fitted-teacher cache differs from live replay")
            state = train_fit(store, data, candidate, device, lambda *_: None, teacher=teacher, bank=bank)
            observed = predictions(store)
            if len(observed) != 72 or not state["complete"]:
                raise RuntimeError("Fixture fit did not complete its full exposure budget")
            from kd_review import audit_fit

            audited = audit_fit(store, data, bank)
            receipt["arms"][arm] = dict(
                updates=state["update"],
                observations=len(observed),
                build_queries=96 if bank is not None else 0,
                independently_reconstructed_loss_error=audited["max_per_image_loss_error"],
            )
    # Corrupt a copy of a selected file and ensure verification fails.
    from shutil import copytree

    bad = root / "corrupt-copy"
    copytree(recovered, bad)
    _, _, broken = unit(bad, "teacher", device)
    with broken.path("config.json").open("ab") as f:
        f.write(b"bad")
    try:
        broken.load()
    except RuntimeError as exc:
        if "Corrupt" not in str(exc):
            raise
    else:
        raise RuntimeError("Corruption was accepted")
    try:
        Checkpoints(recovered, {"incompatible": True})
    except RuntimeError:
        receipt["incompatible_identity_rejected"] = True
    else:
        raise RuntimeError("Incompatible identity accepted")
    receipt["corruption_rejected"] = True
    before = file_info(control.pointer_path)
    child(baseline, device)
    if before != file_info(control.pointer_path):
        raise RuntimeError("Completed fit was rewritten")
    receipt["completed_run_skipped"] = True
    json_write(root / "qualification.json", receipt)
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--worker", type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--until", type=int)
    p.add_argument("--fault", type=int)
    p.add_argument("--qualify", type=Path)
    args = p.parse_args()
    if args.worker:
        resumed_worker(args.worker, args.until, args.fault, args.device)
    else:
        root = args.qualify or Path(tempfile.mkdtemp(prefix="kd-master-check-"))
        print(json.dumps(qualification(root, args.device), indent=2))
