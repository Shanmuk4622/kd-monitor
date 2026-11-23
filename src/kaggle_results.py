"""Release the frozen test cohorts and audit paired results from committed evidence."""

import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from kaggle_run import DEV, HERE, fetch_json, load_input, matrix, publish_files, signature
from kd_fit_engine import evaluate, model_for
from kd_fit_io import csv_write, file_info, hash_object, json_read, json_write, np_write
from kd_review import board_totals, metrics


def energy(h, unit, manifest, revision, base):
    files = [f for f in manifest["files"] if f.startswith("measurement/") and f.endswith("_phase.csv")]
    with tempfile.TemporaryDirectory(dir=base) as tmp:
        h.fetch(unit, manifest, revision, tmp, files)
        return board_totals([Path(tmp) / f for f in files])


def compare(values, method, control):
    seeds = sorted(values)
    differences = [values[s][method] - values[s][control] for s in seeds]
    mean = float(np.mean(differences))
    critical = {3: 4.30265273, 5: 2.77644511}[len(seeds)]
    half = critical * float(np.std(differences, ddof=1)) / math.sqrt(len(seeds))
    return dict(
        seeds=seeds, differences_pp=differences, mean_pp=mean, lower95=mean - half, upper95=mean + half
    )


def release(h, base, out, paths, versions):
    rows = [r for r in matrix() if r["role"] == "student"]
    source = signature()
    selection = {}
    plan = file_info(HERE / "KAGGLE_EXPERIMENT_PLAN.md")["sha256"]
    # Qualify all fit / audit pairs before accessing final labels on any dataset.
    for r in rows:
        run = r["run_id"]
        m, rev = h.selected(run)
        a, ar = h.selected(run + "-review")
        if not m or not m["complete"] or not a or not a["complete"]:
            raise RuntimeError("Final labels remain sealed. Missing fit or audit: " + run)
        if a["identity"] != hash_object(
            {"fit": m["identity"], "chain": m["chain"], "review_source": source}
        ):
            raise ValueError("Audit belongs to a different fit: " + run)
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            h.fetch(run, m, rev, tmp, ["config.json", "ENDPOINT.json"])
            cfg = json_read(Path(tmp) / "config.json")
            ep = json_read(Path(tmp) / "ENDPOINT.json")
        if (
            (cfg["arm"], cfg["model"], cfg["seed"], cfg["epochs"])
            != (r["arm"], r["model"], int(r["seed"]), 60)
            or cfg["contract"]["source"] != source
            or cfg["contract"]["protocol_sha256"] != plan
            or ep["final"] != "epochs/060/weights.pt"
        ):
            raise ValueError("Endpoint identity mismatch: " + run)
        selection[run] = (m, rev, ep["final"], cfg["contract"]["data"])
    results = []
    groups = {}
    costs = {}
    for ds in dict.fromkeys(r["dataset"] for r in rows):
        data, test, clean, _ = load_input(ds, paths, versions, base, release=True)
        teacher_name = f"kaggle-v2.{ds}.resnet34.teacher.907"
        tm, tr = h.selected(teacher_name)
        costs[teacher_name] = energy(h, teacher_name, tm, tr, base)
        for r in (r for r in rows if r["dataset"] == ds):
            run = r["run_id"]
            h.stage = "final evaluation " + run
            m, rev, weights, digest = selection[run]
            if data.identity != digest:
                raise ValueError("Test input differs from the frozen data: " + run)
            name = run + "-final"
            tag = {"model": m["files"][weights]["sha256"], "data": digest, "source": source}
            record = fetch_json(h, name, "metrics.json", base, hash_object(tag))
            if record is None:
                with tempfile.TemporaryDirectory(dir=base) as tmp:
                    tmp = Path(tmp)
                    h.fetch(run, m, rev, tmp, [weights])
                    net = model_for(r["model"], data.classes).to(DEV)
                    net.load_state_dict(torch.load(tmp / weights, map_location=DEV, weights_only=True))
                    z, _, _ = evaluate(net, test, np.arange(len(test.ids)), DEV, 128)
                    del net
                    keep = set(clean.tolist())
                    np_write(tmp / "logits.npy", z)
                    csv_write(
                        tmp / "predictions.csv",
                        ["row", "source", "label", "pred", "eligible"],
                        (
                            (i, k, int(test.labels[i]), int(z[i].argmax()), int(i in keep))
                            for i, k in enumerate(test.ids)
                        ),
                    )
                    record = {}
                    for cohort, indices in (("primary", clean), ("full", np.arange(len(z)))):
                        ok, nll, ece, macro, cm = metrics(z[indices], test.labels[indices], data.classes)
                        record[cohort] = [len(indices), ok, 100 * ok / len(indices), nll, ece, macro]
                        csv_write(
                            tmp / (cohort + "_confusion.csv"),
                            ["true_class", *map(str, range(data.classes))],
                            ((i, *v) for i, v in enumerate(cm)),
                        )
                    json_write(tmp / "metrics.json", record)
                    publish_files(
                        h,
                        name,
                        tmp,
                        tag,
                        [
                            "logits.npy",
                            "predictions.csv",
                            "metrics.json",
                            "primary_confusion.csv",
                            "full_confusion.csv",
                        ],
                    )
            for cohort, v in record.items():
                results.append((run, ds, r["model"], r["seed"], r["arm"], cohort, *v))
            groups.setdefault((ds, r["model"]), {}).setdefault(int(r["seed"]), {})[r["arm"]] = record[
                "primary"
            ][2]
            costs[run] = energy(h, run, m, rev, base)
            if r["bank_action"] != "none":
                cache = f"kaggle-v2.{ds}.cache.{r['arm']}.{r['seed']}"
                if cache not in costs:
                    b, br = h.selected(cache)
                    costs[cache] = energy(h, cache, b, br, base)
    dest = base / "paper_results"
    csv_write(
        dest / "accuracy.csv",
        [
            "run",
            "dataset",
            "model",
            "seed",
            "arm",
            "cohort",
            "n",
            "correct",
            "top1_pct",
            "nll",
            "ece",
            "macro_recall",
        ],
        results,
    )
    paired = {}
    for (ds, model), vals in groups.items():
        p = compare(vals, "disagreement", "shuffled")
        u = compare(vals, "disagreement", "uniform")
        summary = {"vs_shuffled": p, "vs_uniform": u}
        if ds == "cifar100" and model == "resnet18":
            summary["primary_practical_rule_passed"] = (
                p["lower95"] > 0 and u["lower95"] > 0 and p["mean_pp"] >= 0.30
            )
        paired[ds + "/" + model] = summary
    json_write(dest / "paired.json", paired)
    csv_write(
        dest / "resources.csv",
        ["unit", "active_J", "both_J", "wall_s", "windows"],
        ((k, v["active_J"], v["both_reserved_J"], v["wall_s"], v["windows"]) for k, v in costs.items()),
    )
    scenarios = []
    vals = groups["cifar100", "resnet18"]
    teacher = costs["kaggle-v2.cifar100.resnet34.teacher.907"]["active_J"]
    for seed in vals:
        for arm in ("uniform", "shuffled", "entropy", "disagreement"):
            low = compare(vals, arm, "live_kd")["lower95"]
            b = costs[f"kaggle-v2.cifar100.cache.{arm}.{seed}"]["active_J"]
            c = costs[f"kaggle-v2.cifar100.resnet18.{arm}.{seed}"]["active_J"]
            l = costs[f"kaggle-v2.cifar100.resnet18.live_kd.{seed}"]["active_J"]
            for n in (1, 3, 5):
                valid = None not in (b, c, l)
                cached = b + n * c if valid else None
                live = n * l if valid else None
                scenarios.append(
                    (
                        seed,
                        arm,
                        n,
                        cached,
                        live,
                        teacher + cached if valid and teacher is not None else None,
                        teacher + live if valid and teacher is not None else None,
                        math.floor(b / (l - c)) + 1 if valid and l > c else None,
                        low,
                        low > -0.50,
                    )
                )
    csv_write(
        dest / "reuse.csv",
        [
            "seed",
            "arm",
            "students",
            "cache_J",
            "live_J",
            "greenfield_cache_J",
            "greenfield_live_J",
            "first_saving_n",
            "quality_lower95_pp",
            "quality_pass",
        ],
        scenarios,
    )
    files = ["accuracy.csv", "paired.json", "resources.csv", "reuse.csv"]
    rev = publish_files(
        h,
        "paper-results",
        dest,
        {"source": source, "fits": {k: v[0]["chain"] for k, v in selection.items()}},
        files,
    )
    for f in files:
        shutil.copyfile(dest / f, out / f)
    return {"revision": rev, "students_scored": len(rows), "results": str(out)}
