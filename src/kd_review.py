"""Recompute fitted-run evidence and release final CIFAR-100 predictions after all five blocks."""

import copy
import csv
import math
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from cifar100_manifest import RECORD_BYTES
from kd_fit_engine import evaluate, model_for
from kd_fit_io import Checkpoints, csv_write, hash_object, json_read, json_write, np_write


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def audit_fit(store, data, bank=None):
    """Verify committed augmented predictions, loss numerators, full source coverage, selection and denominator."""
    if not store.pointer or not store.pointer["complete"]:
        raise ValueError("A completed fit is required")
    cfg = json_read(store.path("config.json"))
    endpoint = json_read(store.path("ENDPOINT.json"))
    fit_ids = [data.ids[i] for i in data.fit]
    positions = {image_id: i for i, image_id in enumerate(fit_ids)}
    seen = np.zeros((cfg["epochs"], len(fit_ids)), dtype=np.uint8)
    total_obs = 0
    total_updates = 0
    largest = 0.0
    for name in sorted(store.pointer["files"]):
        if not name.endswith("/updates.csv"):
            continue
        folder = store.path(name).parent
        up = read_csv(folder / "updates.csv")
        exposure = read_csv(folder / "exposure.csv")
        logits = np.load(folder / "student_logits.npy", allow_pickle=False)
        if len(exposure) != len(logits):
            raise ValueError("Unmatched training logit rows")
        target = None
        if cfg["arm"] == "live_kd":
            target = np.load(folder / "live_teacher_logits.npy", allow_pickle=False)
        if bank is not None:
            cols = np.array([positions[r["image_id"]] for r in exposure])
            slots = np.array([int(r["bank_slot"]) for r in exposure])
            target = np.array(bank["logits"][bank["offsets"][cols] + slots])
        cursor = 0
        for u in up:
            step = int(u["update"])
            n = int(u["n"])
            if step != total_updates + 1:
                raise ValueError("Missing or repeated update")
            subset = exposure[cursor : cursor + n]
            if len(subset) != n or any(
                int(row["update"]) != step
                or int(row["weight_version"]) != step - 1
                or int(row["logit_row"]) != cursor + j
                for j, row in enumerate(subset)
            ):
                raise ValueError("Exposure cursor mismatch")
            ids = np.asarray([positions[row["image_id"]] for row in subset])
            labels = torch.from_numpy(data.labels[data.fit[ids]])
            if labels.tolist() != [int(row["label"]) for row in subset]:
                raise ValueError("Training labels differ from manifest")
            epoch = int(u["epoch"]) - 1
            np.add.at(seen[epoch], ids, 1)
            z = torch.from_numpy(logits[cursor : cursor + n]).double()
            if int((z.argmax(1) == labels).sum()) != int(u["correct"]):
                raise ValueError("Training accuracy mismatch")
            ce = float(F.cross_entropy(z, labels, reduction="sum"))
            kl = 0.0
            if target is not None:
                t = torch.from_numpy(target[cursor : cursor + n]).double()
                kl = float(F.kl_div((z / 4).log_softmax(1), (t / 4).softmax(1), reduction="sum"))
            elif cfg["arm"] not in ("hard_labels", "teacher"):
                raise ValueError("Cached-target loss audit requires its bank")
            objective = ce if target is None else 0.5 * ce + 8 * kl
            discrepancies = [abs(ce - float(u["ce_sum"])), abs(objective - float(u["objective_sum"]))]
            if target is not None:
                discrepancies.append(abs(kl - float(u["kl_sum"])))
            error = max(discrepancies) / n
            if error > 2e-5:
                raise ValueError(f"Per-example loss reconstruction exceeded tolerance at update {step}")
            largest = max(largest, error)
            cursor += n
            total_updates += 1
        if cursor != len(exposure):
            raise ValueError("Unconsumed training exposure")
        total_obs += cursor
    if not np.all(seen == 1) or total_updates != endpoint["update"]:
        raise ValueError("Incomplete epoch exposure or update ledger")
    by_epoch = []
    for e in range(1, cfg["epochs"] + 1):
        root = store.path(f"epochs/{e:03d}")
        z = torch.from_numpy(np.load(root / "validation_logits.npy", allow_pickle=False)).double()
        ref = read_csv(root / "validation.csv")
        if [r["image_id"] for r in ref] != [data.ids[i] for i in data.validation]:
            raise ValueError("Validation cohort mismatch")
        y = torch.from_numpy(data.labels[data.validation])
        correct = int((z.argmax(1) == y).sum())
        nll = float(F.cross_entropy(z, y))
        m = read_csv(root / "metrics.csv")[0]
        if int(m["validation_correct"]) != correct or abs(nll - float(m["validation_nll"])) > 1e-7:
            raise ValueError("Validation metric reconstruction mismatch")
        by_epoch.append((correct, -nll, -e))
    winner = -max(by_epoch)[2]
    if winner != endpoint["best"]["epoch"]:
        raise ValueError("Validation selection mismatch")
    return dict(
        updates=total_updates,
        observations=total_obs,
        validation_per_epoch=len(data.validation),
        selected_validation_epoch=winner,
        final_epoch=cfg["epochs"],
        max_per_image_loss_error=largest,
        exact_coverage_each_epoch=True,
    )


def metrics(logits, labels, classes):
    z = torch.from_numpy(logits).double()
    y = torch.from_numpy(labels)
    prob = z.softmax(1)
    conf, pred = prob.max(1)
    correct = pred == y
    bins = torch.minimum((15 * conf).long(), torch.full_like(y, 14))
    ece = 0.0
    for b in range(15):
        mask = bins == b
        if mask.any():
            ece += float(mask.double().mean() * abs(correct[mask].double().mean() - conf[mask].mean()))
    confusion = np.bincount(labels * classes + pred.numpy(), minlength=classes**2).reshape(classes, classes)
    supported = confusion.sum(1) > 0
    macro = float(np.mean(confusion.diagonal()[supported] / confusion.sum(1)[supported]))
    return int(correct.sum()), float(F.cross_entropy(z, y)), ece, macro, confusion


def board_totals(files):
    windows = {}
    for path in files:
        for r in read_csv(path):
            windows.setdefault(r["window"], []).append(r)
    active = reserved = wall = 0.0
    active_ok = reserved_ok = bool(windows)
    for rows in windows.values():
        wall += float(rows[0]["wall_s"])
        primary = [r for r in rows if r["active_board"] == "1"]
        if len(primary) == 1 and primary[0]["board_J"] != "":
            active += float(primary[0]["board_J"])
        else:
            active_ok = False
        if len(rows) == 2 and len({r["uuid"] for r in rows}) == 2 and all(r["board_J"] != "" for r in rows):
            reserved += sum(float(r["board_J"]) for r in rows)
        else:
            reserved_ok = False
    return {
        "active_J": active if active_ok else None,
        "both_reserved_J": reserved if reserved_ok else None,
        "wall_s": wall if windows else None,
        "windows": len(windows),
    }


def final_release(hub, base, archive, data, arms, seeds, source, protocol):
    if len(seeds) != 5 or len(arms) != 6:
        raise ValueError("Expected all five predeclared primary blocks and six arms")
    selection = {}
    for s in seeds:
        for arm in arms:
            run = f"v1.cifar100.resnet18.{arm}.{s}"
            manifest, rev = hub.selected(run)
            audit, audit_rev = hub.selected(run + "-review")
            if not manifest or not manifest["complete"] or not audit or not audit["complete"]:
                raise RuntimeError("Final labels remain sealed until this fit and review exist: " + run)
            expected_audit = hash_object(
                {"fit": manifest["identity"], "chain": manifest["chain"], "review_source": source}
            )
            if audit["identity"] != expected_audit:
                raise RuntimeError("The evidence audit does not identify this exact completed fit")
            with tempfile.TemporaryDirectory(dir=base) as temp:
                hub.fetch(run, manifest, rev, temp, ["config.json", "ENDPOINT.json"])
                cfg = json_read(Path(temp) / "config.json")
                ep = json_read(Path(temp) / "ENDPOINT.json")
                if (
                    cfg["contract"]["data"] != data.identity
                    or cfg["contract"]["source"] != source
                    or cfg["contract"]["protocol_sha256"] != protocol
                    or (cfg["seed"], cfg["arm"], cfg["model"]) != (s, arm, "resnet18")
                    or cfg["epochs"] != 60
                    or ep["final"] != "epochs/060/weights.pt"
                ):
                    raise RuntimeError("Model does not match the frozen primary campaign")
            selection[run] = (manifest, rev, ep["final"], audit_rev)
    name = "primary-final-review"
    # Bind stable selected file revisions by content. Unrelated later repository commits do not change identity.
    identity = {
        "data": data.identity,
        "source": source,
        "protocol": protocol,
        "endpoints": {n: m["files"][p]["sha256"] for n, (m, r, p, a) in selection.items()},
    }
    store = Checkpoints(base / "final_review", identity)
    hub.reconcile(name, store)
    if store.pointer and store.pointer["complete"]:
        remote, _ = hub.selected(name)
        if remote is None or remote["chain"] != store.pointer["chain"]:
            store.load()
            hub.publish(name, store)
        return json_read(store.path("paired_result.json"))
    with tarfile.open(archive, "r:gz") as tar:
        with tar.extractfile("cifar-100-binary/test.bin") as stream:
            raw = np.frombuffer(stream.read(), np.uint8).reshape(10000, RECORD_BYTES)
    testing = copy.copy(data)
    testing.rgb = raw[:, 2:].reshape(-1, 3, 32, 32)
    testing.labels = raw[:, 1].astype(np.int64)
    testing.ids = [f"test/{i:05d}" for i in range(len(raw))]
    manifest = read_csv(base / "input" / json_read(base / "input" / "INPUT.json")["manifest"])
    masked = np.array([r["role"] == "test_primary" for r in manifest if r["image_id"].startswith("test/")])
    if len(masked) != len(raw) or not masked.any():
        raise ValueError("Final cohort mismatch")
    rows = []
    files = []
    counts = {}
    for s in seeds:
        for arm in arms:
            run = f"v1.cifar100.resnet18.{arm}.{s}"
            m, rev, weights, audit_rev = selection[run]
            with tempfile.TemporaryDirectory(dir=base) as temp:
                hub.fetch(run, m, rev, temp, [weights])
                model = model_for("resnet18", 100).cuda().eval()
                model.load_state_dict(
                    torch.load(Path(temp) / weights, map_location="cuda:0", weights_only=True)
                )
                z, _, _ = evaluate(model, testing, np.arange(len(raw)), torch.device("cuda:0"), 128)
            pred_folder = f"runs/{run}"
            np_write(store.path(pred_folder + "/logits.npy"), z)
            csv_write(
                store.path(pred_folder + "/predictions.csv"),
                ["row", "image_id", "label", "prediction", "eligible"],
                (
                    (i, testing.ids[i], testing.labels[i], v, int(masked[i]))
                    for i, v in enumerate(z.argmax(1))
                ),
            )
            files.extend([pred_folder + "/logits.npy", pred_folder + "/predictions.csv"])
            for label, mask in (("primary", masked), ("full", np.ones(len(raw), dtype=bool))):
                ok, nll, ece, macro, confusion = metrics(z[mask], testing.labels[mask], 100)
                if label == "primary":
                    counts[s, arm] = ok
                rows.append(
                    (
                        s,
                        arm,
                        label,
                        int(mask.sum()),
                        ok,
                        100 * ok / mask.sum(),
                        nll,
                        ece,
                        macro,
                        rev,
                        audit_rev,
                    )
                )
                path = pred_folder + f"/{label}_confusion.csv"
                csv_write(
                    store.path(path),
                    ["true_class", *[f"pred_{i}" for i in range(100)]],
                    ((i, *line) for i, line in enumerate(confusion)),
                )
                files.append(path)
    csv_write(
        store.path("results.csv"),
        [
            "seed",
            "arm",
            "cohort",
            "n",
            "correct",
            "top1_pct",
            "nll",
            "ece",
            "macro_recall",
            "fit_revision",
            "audit_revision",
        ],
        rows,
    )

    def paired(control, method="disagreement"):
        deltas = np.array([100 * (counts[s, method] - counts[s, control]) / masked.sum() for s in seeds])
        mean = float(deltas.mean())
        radius = float(2.7764451052 * deltas.std(ddof=1) / math.sqrt(5))
        return dict(deltas_pp=deltas.tolist(), mean_pp=mean, lower95=mean - radius, upper95=mean + radius)

    primary, uniform = paired("shuffled"), paired("uniform")
    result = dict(
        denominator=int(masked.sum()),
        seeds=list(seeds),
        guided_minus_shuffled=primary,
        guided_minus_uniform=uniform,
        uniform_test_reached=primary["lower95"] > 0,
        practical_rule_passed=primary["lower95"] > 0
        and uniform["lower95"] > 0
        and primary["mean_pp"] >= 0.3,
    )
    resource = {}
    teacher_unit = "v1.cifar100.resnet34.teacher.907"
    teacher_manifest, teacher_revision = hub.selected(teacher_unit)
    if not teacher_manifest or not teacher_manifest["complete"]:
        raise RuntimeError("Missing completed teacher energy evidence")
    teacher_phases = [
        n for n in teacher_manifest["files"] if n.startswith("measurement/") and n.endswith("_phase.csv")
    ]
    with tempfile.TemporaryDirectory(dir=base) as temp:
        hub.fetch(teacher_unit, teacher_manifest, teacher_revision, temp, teacher_phases)
        resource[teacher_unit] = board_totals([Path(temp) / n for n in teacher_phases])
    for s in seeds:
        for arm in arms:
            fit = f"v1.cifar100.resnet18.{arm}.{s}"
            units = [fit]
            if arm not in ("hard_labels", "live_kd"):
                units.append(f"v1.cifar100.cache.{arm}.{s}")
            for which in units:
                if which in resource:
                    continue
                m, r = (selection[fit][0], selection[fit][1]) if which == fit else hub.selected(which)
                if not m or not m["complete"]:
                    raise RuntimeError("Missing cache energy evidence")
                phase = [n for n in m["files"] if n.startswith("measurement/") and n.endswith("_phase.csv")]
                with tempfile.TemporaryDirectory(dir=base) as temp:
                    hub.fetch(which, m, r, temp, phase)
                    resource[which] = board_totals([Path(temp) / n for n in phase])
    csv_write(
        store.path("resources.csv"),
        ["unit", "active_J", "both_reserved_J", "wall_s", "windows"],
        ((k, v["active_J"], v["both_reserved_J"], v["wall_s"], v["windows"]) for k, v in resource.items()),
    )
    reused = []
    for method in ("uniform", "shuffled", "entropy", "disagreement"):
        quality = paired("live_kd", method)
        for s in seeds:
            b = resource[f"v1.cifar100.cache.{method}.{s}"]["active_J"]
            l = resource[f"v1.cifar100.resnet18.live_kd.{s}"]["active_J"]
            c = resource[f"v1.cifar100.resnet18.{method}.{s}"]["active_J"]
            supported = all(v is not None for v in (b, l, c))
            crossover = max(1, math.floor(b / (l - c)) + 1) if supported and l > c else None
            status = (
                "finite" if crossover else ("no_finite_saving" if supported else "missing_board_reading")
            )
            teacher_cost = resource[teacher_unit]["active_J"]
            for r in (1, 3, 5):
                reused.append(
                    (
                        s,
                        method,
                        r,
                        b,
                        c,
                        l,
                        (b + r * c) if supported else None,
                        r * l if supported else None,
                        teacher_cost,
                        teacher_cost + b + r * c if supported and teacher_cost is not None else None,
                        teacher_cost + r * l if supported and teacher_cost is not None else None,
                        crossover,
                        status,
                        quality["lower95"],
                        quality["lower95"] > -0.5,
                    )
                )
    csv_write(
        store.path("reuse_scenarios.csv"),
        [
            "seed",
            "method",
            "students_scenario",
            "cache_build_J",
            "cached_fit_J",
            "live_fit_J",
            "projected_cached_total_J",
            "projected_live_total_J",
            "teacher_training_J",
            "greenfield_cached_J",
            "greenfield_live_J",
            "first_strict_saving_n",
            "energy_status",
            "paired_accuracy_lower95_pp",
            "quality_margin_pass",
        ],
        reused,
    )
    json_write(store.path("paired_result.json"), result)
    store.commit(
        {"update": 30, "complete": True},
        files + ["results.csv", "paired_result.json", "resources.csv", "reuse_scenarios.csv"],
    )
    hub.publish(name, store)
    return result
