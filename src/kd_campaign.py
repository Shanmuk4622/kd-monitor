"""Sequential CIFAR-100 teacher and six-arm runner for a reserved Kaggle session."""

import csv
import hashlib
import os
import shutil
import signal
import sys
import tempfile
import time
import uuid
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from filelock import FileLock

from kd_fit_data import build_bank, official_images
from kd_fit_engine import model_for, train_fit
from kd_review import audit_fit, final_release
from kd_fit_io import Checkpoints, csv_write, file_info, hash_object, json_read, json_write
from kd_hub import Hub
from kd_meter import BoardWindow
from kd_qualify import child, qualification, unit, verify_continuation

ARMS = ("hard_labels", "live_kd", "uniform", "shuffled", "entropy", "disagreement")
SEEDS = (42, 123, 2026, 31415, 27182)
RESERVE = 2_000_000_000


def environment():
    import importlib.metadata as md

    return dict(
        python=sys.version.split()[0],
        torch=str(torch.__version__),
        numpy=np.__version__,
        torchvision=md.version("torchvision"),
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        capability=[list(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())],
    )


def qualified_source():
    root = Path(__file__).resolve().parent
    paths = [
        "cache_preflight.py",
        "cifar100_manifest.py",
        "kd_fit_io.py",
        "kd_fit_data.py",
        "kd_fit_engine.py",
        "kd_hub.py",
        "kd_meter.py",
        "kd_campaign.py",
        "kd_qualify.py",
        "kd_review.py",
    ]
    return hash_object({n: file_info(root / n)["sha256"] for n in paths})


def footprint_probe(data, base, model_name, epochs, device, on_commit=None):
    cfg = dict(
        seed=119,
        arm="teacher" if model_name == "resnet34" else "live_kd",
        epochs=epochs,
        batch=128,
        warm_epochs=5,
        commit_updates=32,
        model=model_name,
    )
    teacher = model_for("resnet34", data.classes).to(device).eval() if cfg["arm"] == "live_kd" else None
    store = Checkpoints(
        base, {"fixture": "capacity", "cfg": cfg, "data": data.identity, "environment": environment()}
    )
    steps = (len(data.fit) + 127) // 128
    start = time.perf_counter()
    state = train_fit(
        store, data, cfg, device, on_commit or (lambda *_: None), teacher=teacher, until=steps
    )
    elapsed = time.perf_counter() - start
    paths = store.pointer["files"]
    if state["update"] != steps:
        raise RuntimeError("The capacity probe must cover one full fitting epoch")
    families = {}
    step_times = []
    for name, metadata in paths.items():
        family = name.split("/", 1)[0]
        totals = families.setdefault(family, {"files": 0, "bytes": 0})
        totals["files"] += 1
        totals["bytes"] += metadata["bytes"]
        if name.endswith("/updates.csv"):
            with store.path(name).open(encoding="utf-8", newline="") as f:
                step_times.extend(float(row["seconds"]) for row in csv.DictReader(f))
    state_bytes = paths[store.pointer["state_file"]]["bytes"]
    per_epoch = sum(families[family]["bytes"] for family in ("updates", "epochs", "ledger"))
    # Three rolling-state bodies cover the selected generation, fallback and temporary write.
    need = int(1.20 * (epochs * per_epoch + 3 * state_bytes) + 400_000_000)
    if len(step_times) != steps:
        raise RuntimeError("Missing per-update timings in the capacity probe")
    seconds_per_update = float(np.median(step_times))
    return dict(
        model=model_name,
        epochs=epochs,
        observed_updates=steps,
        observed_examples=len(data.fit),
        observed_validation_examples=len(data.validation),
        observed_epoch_families=families,
        rolling_state_bytes=state_bytes,
        required_incremental_bytes=need,
        working_reserve=RESERVE,
        probe_elapsed_s=elapsed,
        measured_step_seconds=seconds_per_update,
        compute_only_forecast_hours=seconds_per_update * steps * epochs / 3600,
        local_epoch_inclusive_forecast_hours=elapsed * epochs / 3600,
        full_training_updates=steps * epochs,
        service_time_in_forecast=False,
    )


def erase_verified_copies(store, manifest):
    """Only disposable Kaggle working copies already verified in the immutable remote revision."""
    if not any(
        p.resolve() in store.root.resolve().parents
        for p in (Path("/kaggle/working"), Path("/kaggle/temp/kd_view_v2"))
    ):
        return
    keep = {"ENDPOINT.json", "config.json", "POINTER.json"}
    for name, item in manifest["files"].items():
        if name in keep:
            continue
        path = store.path(name)
        if path.is_file() and file_info(path) == {k: item[k] for k in ("bytes", "sha256")}:
            path.unlink()
    # The unselected rolling slot is superseded by the verified complete fit.
    for name in ("rolling/state_0.pt", "rolling/state_1.pt"):
        path = store.path(name)
        if name not in manifest["files"] and path.is_file():
            path.unlink()


class Driver:
    def __init__(self, base, hub, data, device, archives, receipt):
        self.base, self.hub, self.data, self.device = base, hub, data, device
        self.archives, self.receipt = archives, receipt
        self.stopping = False
        self.source = qualified_source()

    def run_unit(self, name, config, required, work, keep=()):
        store = Checkpoints(
            self.base / "units" / name,
            {
                "source": self.source,
                "config": config,
                "data": self.data.identity,
                "environment": environment(),
                "protocol_sha256": self.receipt["protocol_sha256"],
            },
        )
        remote, rev = self.hub.selected(name)
        if remote and remote["identity"] != store.tag:
            raise RuntimeError("Previous unit binds different data, code, teacher or environment")
        if remote and remote["complete"]:
            if keep:
                self.hub.fetch(name, remote, rev, store.root, keep)
            print("Verified completed unit, skipping", name, rev, flush=True)
            return store, remote, rev
        self.hub.reconcile(name, store, selected=(remote, rev))
        live_files = sum(p.stat().st_size for p in store.root.rglob("*") if p.is_file())
        free = shutil.disk_usage(self.base).free
        working = Path("/kaggle/working")
        logical = (
            sum(p.stat().st_size for p in working.rglob("*") if p.is_file()) if working.is_dir() else 0
        )
        available = (
            min(free, 20_000_000_000 - logical)
            if working.resolve() in self.base.resolve().parents
            else free
        )
        if available < max(0, required - live_files) + RESERVE:
            raise RuntimeError(
                f"Storage projection failed for {name}. Free or quota bytes {available}, projected {required}."
            )
        setup_dir = store.root / "setup_telemetry"
        idle = BoardWindow(setup_dir, "pre_run_idle", self.device).start()
        time.sleep(3)
        idle.finish()
        warm = BoardWindow(setup_dir, "standard_warmup", self.device).start()
        with torch.inference_mode():
            example = torch.ones((768, 768), device=self.device)
            for _ in range(20):
                torch.mm(example, example)
            del example
        warm.finish()
        last = time.monotonic()
        window = None

        def begin():
            return BoardWindow(store.root / "measurement", name, self.device).start()

        def publish():
            nonlocal window, last
            if window is not None:
                window.finish()
                window = None
            if store.pointer is not None:
                extras = {
                    p.relative_to(store.root).as_posix(): file_info(p)
                    for folder in ("measurement", "setup_telemetry")
                    for p in (store.root / folder).glob("*.csv")
                }
                self.hub.publish(name, store, extras)
            last = time.monotonic()

        def hook(current, force):
            nonlocal window
            queue = getattr(self.hub, "queue", None)
            if queue and queue.lost:
                raise queue.lost
            if time.monotonic() >= getattr(self, "deadline", float("inf")):
                self.stopping = True
            if force or self.stopping or time.monotonic() - last >= 1800:
                if current.pointer["complete"] and window is not None:
                    window.finish()
                    window = None
                    cooldown = BoardWindow(setup_dir, "post_run_idle", self.device).start()
                    time.sleep(3)
                    cooldown.finish()
                publish()
                if self.stopping:
                    raise KeyboardInterrupt("Catchable stop at durable update")
                if not current.pointer["complete"]:
                    window = begin()

        try:
            window = begin()
            work(store, hook)
            if window is not None:
                publish()
        except BaseException:
            try:
                publish()
            except Exception as transport:
                print("Local files retained after publication error", type(transport).__name__, flush=True)
            raise
        manifest, revision = self.hub.selected(name)
        if not manifest or not manifest["complete"] or manifest["chain"] != store.pointer["chain"]:
            raise RuntimeError("Completed unit lacks a matching verified remote selection")
        return store, manifest, revision

    def fit(self, config, budget, teacher=None, teacher_id=None, bank=None):
        arm, seed = config["arm"], config["seed"]
        prefix = getattr(self.data, "prefix", "v1.cifar100")
        name = f"{prefix}.{config['model']}.{arm}.{seed}"
        binding = {**config, "teacher_sha256": teacher_id, "bank": bank["identity"] if bank else None}

        def work(store, hook):
            if teacher is not None:
                teacher.to(self.device)
            try:
                return train_fit(
                    store, self.data, config, self.device, hook, lambda: self.stopping, teacher, bank
                )
            finally:
                if teacher is not None:
                    teacher.cpu()

        store, manifest, rev = self.run_unit(name, binding, budget, work, keep=["ENDPOINT.json"])
        endpoint = json_read(store.path("ENDPOINT.json"))
        review_name = name + "-review"
        review = Checkpoints(
            self.base / "reviews" / name,
            {"fit": store.tag, "chain": manifest["chain"], "review_source": self.source},
        )
        old, review_rev = self.hub.selected(review_name)
        if old and old["identity"] != review.tag:
            raise RuntimeError("Previous evidence audit has different fit identity")
        if not old or not old["complete"]:
            if store.pointer is None or store.pointer["chain"] != manifest["chain"]:
                self.hub.fetch(
                    name,
                    manifest,
                    rev,
                    store.root,
                    [n for n in manifest["files"] if n != "POINTER.json"] + ["POINTER.json"],
                )
                store.pointer = json_read(store.pointer_path)
            audit = audit_fit(store, self.data, bank)
            json_write(review.path("fit_audit.json"), audit)
            review.commit(dict(update=audit["updates"], complete=True), ["fit_audit.json"])
            review_rev = self.hub.publish(review_name, review)
        if arm == "teacher":
            best = endpoint["best"]["weights"]
            self.hub.fetch(name, manifest, rev, store.root, [best])
            result = model_for(config["model"], self.data.classes).cpu().eval().requires_grad_(False)
            result.load_state_dict(torch.load(store.path(best), map_location="cpu", weights_only=True))
            identity = manifest["files"][best]["sha256"]
        else:
            result, identity = None, None
        self.archives[name] = {
            "revision": rev,
            "review_revision": review_rev,
            "endpoint": endpoint,
            "identity": store.tag,
        }
        json_write(self.base / "run_registry.json", self.archives)
        erase_verified_copies(store, manifest)
        return result, identity

    def bank(self, seed, arm, teacher, teacher_id):
        prefix = getattr(self.data, "prefix", "v1.cifar100")
        name = f"{prefix}.cache.{arm}.{seed}"
        cfg = {"seed": seed, "policy": arm, "teacher_sha256": teacher_id}

        def work(store, hook):
            try:
                build_bank(store, self.data, teacher.to(self.device), self.device, seed, arm, hook)
            finally:
                teacher.cpu()

        keep = ["bank/codes.npy", "bank/counts.npy", "bank/logits.npy", "bank/scores.csv"]
        store, manifest, rev = self.run_unit(
            name, cfg, max(700_000_000, 16 * len(self.data.fit) * self.data.classes * 3), work, keep=keep
        )
        from cache_preflight import key

        counts = np.load(store.path(keep[1]), allow_pickle=False)
        bank = dict(
            counts=counts,
            offsets=np.cumsum(counts, dtype=np.int64) - counts,
            codes=np.load(store.path(keep[0]), allow_pickle=False),
            logits=np.load(store.path(keep[2]), mmap_mode="r", allow_pickle=False),
            slot_offsets=np.array(
                [key("slot-offset", seed, self.data.ids[i]) % int(c) for i, c in zip(self.data.fit, counts)]
            ),
            identity=hash_object({"contract": store.tag, "logits": manifest["files"][keep[2]]["sha256"]}),
        )
        gate_name = name + "-replay"
        gate = Checkpoints(
            self.base / "reviews" / gate_name, {"fit_bank": bank["identity"], "source": self.source}
        )
        previous, _ = self.hub.selected(gate_name)
        if previous and previous["identity"] != gate.tag:
            raise RuntimeError("A different replay audit already exists")
        if not previous or not previous["complete"]:
            import copy
            from cache_preflight import kd_loss

            secondary = torch.device("cuda:1")
            anchor = copy.deepcopy(teacher).to(secondary).eval()
            pos = np.random.default_rng(seed).choice(len(counts), 128, replace=False)
            slots = counts[pos] - 1
            x, y = self.data.inputs(self.data.fit[pos], bank["codes"][pos, slots], secondary)
            expected = torch.from_numpy(np.array(bank["logits"][bank["offsets"][pos] + slots])).to(
                secondary
            )
            with torch.no_grad():
                actual = anchor(x)
            student = model_for("resnet18", self.data.classes).to(secondary).eval()
            z = student(x)
            a, b = kd_loss(z, actual, y), kd_loss(z, expected, y)
            g1 = torch.autograd.grad(a, tuple(student.parameters()), retain_graph=True)
            g2 = torch.autograd.grad(b, tuple(student.parameters()))
            error = dict(
                logit=(actual - expected).abs().max().item(),
                loss=abs(a.item() - b.item()),
                gradient=max((i - j).abs().max().item() for i, j in zip(g1, g2)),
                audit_queries=128,
            )
            if error["logit"] > 1e-5 or error["loss"] > 1e-6 or error["gradient"] > 1e-5:
                json_write(gate.path("failed_replay.json"), error)
                raise RuntimeError("Actual trained teacher failed the cross-device replay gate")
            json_write(gate.path("replay.json"), error)
            gate.commit({"update": 128, "complete": True}, ["replay.json"])
            self.hub.publish(gate_name, gate)
        if arm == "disagreement" and self.data.classes == 100:
            diagnostic_name = name + "-independent-views"
            extra = Checkpoints(
                self.base / "reviews" / diagnostic_name, {"bank": bank["identity"], "source": self.source}
            )
            present, _ = self.hub.selected(diagnostic_name)
            if present and present["identity"] != extra.tag:
                raise RuntimeError("Independent-views diagnostic identity differs")
            if not present or not present["complete"]:
                from cache_preflight import pilot_scores
                from scipy.stats import spearmanr
                from kd_review import read_csv

                fitting_labels = self.data.labels[self.data.fit]
                original = read_csv(store.path("bank/scores.csv"))
                count_codes = 2 * (2 * self.data.padding + 1) ** 2
                records = []
                rho = []
                teacher.to(self.device)
                for label in range(self.data.classes):
                    members = np.flatnonzero(fitting_labels == label)
                    subset = sorted(
                        members, key=lambda pos: key("holdout-probe-ids", self.data.ids[self.data.fit[pos]])
                    )[:10]
                    pairs = []
                    for pos in subset:
                        options = np.setdiff1d(np.arange(count_codes), bank["codes"][pos])
                        rng = np.random.default_rng(
                            key("holdout-probe-views", seed, self.data.ids[self.data.fit[pos]])
                        )
                        pairs.append(rng.choice(options, 2, replace=False))
                    codes = np.asarray(pairs)
                    x, _ = self.data.inputs(self.data.fit[np.repeat(subset, 2)], codes.ravel(), self.device)
                    with torch.inference_mode():
                        js, _ = pilot_scores(teacher(x).reshape(len(subset), 2, -1).cpu())
                    baseline = [float(original[pos]["js_nats"]) for pos in subset]
                    corr = float(spearmanr(baseline, js).statistic)
                    rho.append(corr)
                    records.extend(
                        (seed, label, self.data.ids[self.data.fit[pos]], q[0], q[1], baseline[j], js[j])
                        for j, (pos, q) in enumerate(zip(subset, pairs))
                    )
                csv_write(
                    extra.path("independent_views.csv"),
                    [
                        "seed",
                        "label",
                        "image_id",
                        "new_code_0",
                        "new_code_1",
                        "ranking_js_nats",
                        "independent_js_nats",
                    ],
                    records,
                )
                csv_write(
                    extra.path("class_rank_associations.csv"),
                    ["class", "spearman_rho"],
                    ((i, r if np.isfinite(r) else "") for i, r in enumerate(rho)),
                )
                extra.commit(
                    {"update": 2 * len(records), "complete": True},
                    ["independent_views.csv", "class_rank_associations.csv"],
                )
                self.hub.publish(diagnostic_name, extra)
                teacher.cpu()
        self.archives[name] = {"revision": rev, "identity": store.tag}
        return bank, store, manifest


def run_master(mode, base, archive=None, repo=None, blocks=(42,), protocol_sha256=None):
    base = Path(base).resolve()
    base.mkdir(parents=True, exist_ok=True)
    if mode not in ("qualification", "measured", "final_review"):
        raise ValueError("Choose qualification, measured or final_review mode")
    if mode == "qualification":
        return {
            str(device): qualification(Path(tempfile.mkdtemp(prefix="local-gate-", dir=base)), device)
            for device in (["cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())])
        }
    if not set(blocks).issubset(SEEDS) or len(set(blocks)) != len(blocks) or not blocks:
        raise ValueError("Choose distinct preregistered seed blocks")
    if not archive or not repo or not protocol_sha256:
        raise ValueError(
            "Measured work requires the original binary archive, a dedicated HF dataset and protocol hash"
        )
    env = environment()
    if (
        len(env["gpus"]) != 2
        or any("T4" not in n for n in env["gpus"])
        or not Path("/kaggle/working").is_dir()
    ):
        raise RuntimeError("Measured qualification requires a Kaggle dual-T4 session")
    from kaggle_secrets import UserSecretsClient

    hub = Hub(repo, UserSecretsClient().get_secret("HF_TOKEN"), base / "transport.jsonl")
    seedset = tuple(blocks)
    with FileLock(str(base / "owner.lock"), timeout=1):
        hub.claim()
        driver = None
        previous = {}
        try:
            data = official_images(archive, base / "input")
            if mode == "final_review":
                return final_release(
                    hub, base, archive, data, ARMS, SEEDS, qualified_source(), protocol_sha256
                )
            gate = Path(tempfile.mkdtemp(prefix="target-gate-", dir=base))
            # Both cards explicitly perform replay/restart work. Scientific timing occupies GPU 0 alone.
            first = qualification(gate / "gpu0", "cuda:0", all_arms=True)
            second = qualification(gate / "gpu1", "cuda:1", all_arms=False)
            dev = torch.device("cuda:0")
            name = "qualification-" + uuid.uuid4().hex
            child(gate / "seam_source", dev, until=2)
            _, _, probe = unit(gate / "seam_source", "teacher", dev)
            rev = hub.publish(name, probe)
            snapshot = hub.read(name + "/MANIFEST.json", rev)
            recovered = gate / "seam_download"
            hub.fetch(name, snapshot, rev, recovered)
            child(recovered, dev)
            seam = verify_continuation(gate / "gpu0" / "uninterrupted", recovered, dev)
            forecasts = {}
            for model, epochs in (("resnet34", 90), ("resnet18", 60)):
                with tempfile.TemporaryDirectory(prefix="capacity-" + model + "-", dir=gate) as disposable:
                    forecasts[model] = footprint_probe(data, Path(disposable), model, epochs, dev)
            receipt = dict(
                protocol_sha256=protocol_sha256,
                source=qualified_source(),
                environment=env,
                data=data.identity,
                first=first,
                second=second,
                remote_gate_revision=rev,
                pinned_download_continuation=seam,
                forecasts=forecasts,
            )
            json_write(gate / "target_qualification.json", receipt)
            # Preserve the successful on-target receipts and split before any measured fit.
            gate_store = Checkpoints(
                gate / "release",
                {"source": qualified_source(), "dataset": data.identity, "kind": "qualification"},
            )
            from shutil import copyfile

            names = ["target_qualification.json", "split_manifest.csv", "split_receipt.json", "protocol.md"]
            input_record = json_read(base / "input" / "INPUT.json")
            split = base / "input" / input_record["manifest"]
            for src, dst in zip(
                [
                    gate / "target_qualification.json",
                    split,
                    split.with_name("split_receipt.json"),
                    Path(__file__).with_name("PROTOCOL.md"),
                ],
                names,
            ):
                copyfile(src, gate_store.path(dst))
            gate_store.commit(dict(update=0, complete=True), names)
            hub.publish(name + "-evidence", gate_store)
            archives = (
                json_read(base / "run_registry.json") if (base / "run_registry.json").exists() else {}
            )
            driver = Driver(base, hub, data, dev, archives, receipt)

            def signal_stop(sig, frame):
                driver.stopping = True
                print("Stop requested. Finishing the current complete update.", flush=True)

            for sig in (signal.SIGINT, signal.SIGTERM):
                previous[sig] = signal.signal(sig, signal_stop)
            teacher_cfg = dict(
                seed=907,
                arm="teacher",
                model="resnet34",
                epochs=90,
                batch=128,
                warm_epochs=5,
                commit_updates=32,
            )
            teacher, teacher_id = driver.fit(
                teacher_cfg, forecasts["resnet34"]["required_incremental_bytes"]
            )
            for seed in seedset:
                order = sorted(
                    ARMS,
                    key=lambda a: hashlib.sha256(
                        f"order-v1|cifar100|resnet18|{seed}|{a}".encode()
                    ).digest(),
                )
                for arm in order:
                    cfg = dict(
                        seed=seed,
                        arm=arm,
                        model="resnet18",
                        epochs=60,
                        batch=128,
                        warm_epochs=5,
                        commit_updates=32,
                    )
                    bank = bank_store = bank_manifest = None
                    if arm not in ("hard_labels", "live_kd"):
                        bank, bank_store, bank_manifest = driver.bank(seed, arm, teacher, teacher_id)
                    driver.fit(
                        cfg,
                        forecasts["resnet18"]["required_incremental_bytes"],
                        teacher if arm == "live_kd" else None,
                        teacher_id if arm != "hard_labels" else None,
                        bank,
                    )
                    if bank_store:
                        del bank
                        erase_verified_copies(bank_store, bank_manifest)
            return {
                "qualified": True,
                "data_sha256": data.identity,
                "registry": driver.archives,
                "test_release": "sealed",
                "qualification_revision": rev,
            }
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            hub.release()
