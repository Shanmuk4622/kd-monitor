"""Disposable checks for the KD view-allocation study. No benchmark data or fitted weights."""

import csv
import hashlib
import json
import math
import os
import platform
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

# cuBLAS reads this at initialization in a fresh process or notebook kernel.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn
from torch.nn import functional as F


POLICIES = ("uniform", "shuffled", "entropy", "disagreement")


def key(*fields):
    """Stable, domain-separated 64-bit seed for a local random generator."""
    raw = json.dumps(fields, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")


def view_codes(image_id, seed, padding):
    count = 2 * (2 * padding + 1) ** 2
    rng = np.random.Generator(np.random.PCG64(key("view-bank", seed, image_id)))
    return rng.permutation(count)[:6].tolist()


def render_view(image, code, padding):
    """Decode one crop and flip from CHW RGB. The caller owns normalization."""
    width = 2 * padding + 1
    crop, flip = divmod(code, 2)
    top, left = divmod(crop, width)
    if not (0 <= top < width):
        raise ValueError("Invalid crop code")
    height, size = image.shape[-2:]
    padded = F.pad(image.unsqueeze(0), (padding,) * 4, mode="reflect")[0]
    view = padded[:, top:top + height, left:left + size]
    return view.flip(-1) if flip else view


def pilot_scores(logits):
    """Two-view Jensen-Shannon divergence and mean individual entropy in nats."""
    if logits.ndim != 3 or logits.shape[1] != 2 or not torch.isfinite(logits).all():
        raise ValueError("Expected finite N by 2 by C pilot logits")
    logp = logits.double().log_softmax(-1)
    logm = torch.logaddexp(logp[:, 0], logp[:, 1]) - math.log(2)
    prob = logp.exp()
    entropy = -(prob * logp).sum(-1).mean(1)
    js = (prob * (logp - logm[:, None, :])).sum(-1).mean(1)
    return js.cpu().numpy(), entropy.cpu().numpy()


def allocate(ids, labels, scores, policy, seed):
    """For an odd class, its median receives four views to preserve the exact budget."""
    if policy not in POLICIES or len(set(ids)) != len(ids):
        raise ValueError("Unknown allocation or duplicate image IDs")
    if len(ids) != len(labels) or len(ids) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Invalid allocation inputs")
    counts = {image_id: 4 for image_id in ids}
    if policy == "uniform":
        return counts
    for label in sorted(set(labels)):
        members = [i for i, value in enumerate(labels) if value == label]
        if policy == "shuffled":
            members.sort(key=lambda i: (key("random-high", seed, ids[i]), ids[i]))
        else:
            members.sort(key=lambda i: (-float(scores[i]), key("tie", seed, ids[i]), ids[i]))
        half = len(members) // 2
        for rank, index in enumerate(members):
            counts[ids[index]] = 6 if rank < half else (2 if rank >= len(members) - half else 4)
    return counts


def slot_for_epoch(image_id, seed, epoch, count):
    return (key("slot-offset", seed, image_id) + epoch) % count


def kd_loss(student, teacher, labels, temperature=4.0, weight=0.5):
    ce = F.cross_entropy(student, labels)
    kl = F.kl_div((student / temperature).log_softmax(-1),
                  (teacher.detach() / temperature).softmax(-1), reduction="batchmean")
    return (1 - weight) * ce + weight * temperature ** 2 * kl


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def exercise_device(device, output):
    output.mkdir()
    seed = 42
    # An odd class exercises the median case alongside two even classes.
    labels = [0] * 8 + [1] * 8 + [2] * 9
    ids = [f"fit/{i:03d}" for i in range(len(labels))]
    held_out = {"validation/000", "test/000"}
    rng = np.random.default_rng(794)
    images = {i: torch.from_numpy(rng.integers(0, 256, (3, 16, 16), dtype=np.uint8))
              for i in ids}
    codes = {i: view_codes(i, seed, padding=2) for i in ids}
    require(all(len(set(values)) == 6 for values in codes.values()), "Repeated view key")
    torch.manual_seed(827)
    teacher = nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8), nn.ReLU(),
                            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(), nn.Linear(32, 3))
    teacher = teacher.to(device).eval().requires_grad_(False)
    student = nn.Sequential(nn.Conv2d(3, 5, 3), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                            nn.Flatten(), nn.Linear(5, 3)).to(device)

    def inputs(keys):
        return torch.stack([render_view(images[i], codes[i][j], 2).float().div(255)
                            for i, j in keys]).to(device)

    query_counts = Counter()

    def query(keys, phase):
        query_counts[phase] += len(keys)
        with torch.no_grad():
            return teacher(inputs(keys)).cpu()

    probes = [(i, j) for i in ids for j in range(2)]
    pilot = query(probes, "shared_pilot").reshape(len(ids), 2, -1)
    js, entropy = pilot_scores(pilot)
    equal_js, _ = pilot_scores(pilot[:, :1].expand(-1, 2, -1))
    require(np.max(np.abs(equal_js)) < 1e-12, "Identical probe distributions disagreed")
    require(np.all((js >= -1e-12) & (js <= math.log(2) + 1e-12)), "JSD out of range")
    scores = [{"image_id": i, "label": y, "js_nats": j, "mean_entropy_nats": h}
              for i, y, j, h in zip(ids, labels, js, entropy)]
    write_csv(output / "pilot_scores.csv", scores)
    stats = []
    for policy in POLICIES:
        counts = allocate(ids, labels, entropy if policy == "entropy" else js, policy, seed)
        for label in set(labels):
            members = [i for i, y in zip(ids, labels) if y == label]
            require(sum(counts[i] for i in members) == 4 * len(members), "Class budget mismatch")
        extra = [(i, j) for i in ids for j in range(2, counts[i])]
        keys = probes + extra
        require(len(set(keys)) == 4 * len(ids), "Duplicate or missing cache rows")
        require({i for i, _ in keys} == set(ids) and held_out.isdisjoint(i for i, _ in keys),
                "Non-fitting ID in cache")
        values = torch.cat([pilot.reshape(len(probes), -1), query(extra, policy)]).numpy()
        charged = query_counts["shared_pilot"] + query_counts[policy]
        require(charged == 4 * len(ids), "Teacher call ledger differs from cache budget")
        bank = output / f"{policy}_logits.npy"
        index = output / f"{policy}_views.csv"
        np.save(bank, values, allow_pickle=False)
        write_csv(index, [{"row": r, "image_id": i, "slot": j, "view_code": codes[i][j]}
                          for r, (i, j) in enumerate(keys)])
        with index.open(encoding="utf-8", newline="") as f:
            restored_index = list(csv.DictReader(f))
        lookup = {(row["image_id"], int(row["slot"])): int(row["row"]) for row in restored_index}
        cache = np.load(bank, allow_pickle=False)
        require(len(lookup) == len(cache) == len(keys), "Index/body mismatch")
        # Rebuild every stored view independently from the persisted crop code.
        replay_images = torch.stack([
            render_view(images[row["image_id"]], int(row["view_code"]), 2).float().div(255)
            for row in restored_index]).to(device)
        with torch.no_grad():
            live_all = teacher(replay_images).cpu().numpy()
        logit_error = float(np.max(np.abs(live_all - cache)))
        sample_indices = list(range(0, 24, 3))
        selected = [(ids[k], slot_for_epoch(ids[k], seed, 17, counts[ids[k]])) for k in sample_indices]
        cached = torch.from_numpy(cache[[lookup[item] for item in selected]]).to(device)
        batch = inputs(selected)
        target = torch.tensor([labels[k] for k in sample_indices], device=device)
        with torch.no_grad():
            live = teacher(batch)
        pred = student(batch)
        online_loss, cached_loss = kd_loss(pred, live, target), kd_loss(pred, cached, target)
        grad_live = torch.autograd.grad(online_loss, tuple(student.parameters()), retain_graph=True)
        grad_cache = torch.autograd.grad(cached_loss, tuple(student.parameters()))
        grad_error = max((a - b).abs().max().item() for a, b in zip(grad_live, grad_cache))
        loss_error = abs(online_loss.item() - cached_loss.item())
        require(logit_error <= 1e-5 and loss_error <= 1e-6 and grad_error <= 1e-5,
                f"Replay mismatch for {policy}: logits {logit_error}, loss {loss_error}, grad {grad_error}")
        require(all(p.grad is None for p in teacher.parameters()), "Teacher gradient")
        for i in ids:
            exposure = Counter(slot_for_epoch(i, seed, e, counts[i]) for e in range(60))
            require(len(exposure) == counts[i] and set(exposure.values()) == {60 // counts[i]},
                    "Unequal slot exposure across the planned 60 epochs")
        stats.append(dict(policy=policy, fit_images=len(ids), pilot_queries=query_counts["shared_pilot"],
                          additional_queries=query_counts[policy], standalone_queries=charged,
                          replay_audit_queries=len(restored_index) + len(selected),
                          logit_payload_bytes=cache.nbytes, index_bytes=index.stat().st_size,
                          max_logit_error=logit_error, loss_error=loss_error,
                          max_gradient_error=grad_error, bank_sha256=digest(bank),
                          index_sha256=digest(index)))
    write_csv(output / "policy_checks.csv", stats)
    return stats


def run_fixture(output_parent):
    """Make a new evidence directory on every invocation and preserve previous receipts."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    output_parent = Path(output_parent)
    output_parent.mkdir(parents=True, exist_ok=True)
    target = Path(tempfile.mkdtemp(prefix="cache-fixture-", dir=output_parent))
    devices = [torch.device("cpu")] + [torch.device("cuda", n) for n in range(torch.cuda.device_count())]
    results = {}
    for device in devices:
        tag = str(device).replace(":", "-")
        results[tag] = exercise_device(device, target / tag)
    receipt = dict(kind="synthetic engineering fixture", torch=torch.__version__,
                   numpy=np.__version__, python=platform.python_version(), devices=[
                       torch.cuda.get_device_name(n) for n in range(torch.cuda.device_count())
                   ], results=results)
    (target / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return target, receipt
