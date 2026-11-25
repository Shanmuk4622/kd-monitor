"""Kaggle-only execution of the frozen matrix with independently fenced HF publishers."""

import csv
from copy import copy
import importlib.metadata as md
import shutil
import signal
import sys
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from filelock import FileLock, Timeout
from kaggle_inputs import SOURCES, load_dataset, mount
from kaggle_plan import worker_plan
from kd_campaign import Driver, environment, erase_verified_copies, footprint_probe
from kd_fit_io import Checkpoints, canonical, file_info, hash_object, json_read, json_write
from kd_hub import Hub, WriterLeaseBusy
from kd_fit_engine import model_for
from kd_qualify import qualification, child, unit, verify_continuation

HERE = Path(__file__).parent
PREVIOUS_PREFIX = "kd-view-budget/four-workers-v1"
PREFIX = "kd-view-budget/four-workers-tinyimagenet200-v1"
DEV = torch.device("cuda:0")


def matrix():
    with (HERE / "RUN_MATRIX_V2.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def signature():
    files = sorted(
        [
            *HERE.glob("*.py"),
            HERE / "KAGGLE_EXPERIMENT_PLAN.md",
            HERE / "PROTOCOL.md",
            HERE / "RUN_MATRIX_V2.csv",
        ]
    )
    actual = hash_object({p.name: file_info(p)["sha256"] for p in files})
    mapping = HERE / "CAMPAIGN_COMPATIBILITY.json"
    return json_read(mapping).get(actual, actual) if mapping.exists() else actual


class Publisher(Hub):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.guard = threading.RLock()
        self.stage = "opening"
        self.problem = None

    def claim(self):
        while True:
            try:
                return super().claim()
            except WriterLeaseBusy as busy:
                delay = min(300, max(1, busy.expires - time.time()))
                print(f"{busy}. Checking again in {delay:.0f}s. "
                      "If another copy is running, you can cancel this waiting copy.", flush=True)
                time.sleep(delay)

    def write_commit(self, *args, **kwargs):
        with self.guard:
            if self.problem:
                raise RuntimeError("HF heartbeat failed") from self.problem
            revision = super().write_commit(*args, **kwargs)
            queue = getattr(self, "queue", None)
            if queue and queue.job and args and queue.job["run"] + "/MANIFEST.json" in args[0]:
                queue.call("checkpoint", queue.job)
            return revision

    def pulse(self, done):
        next_upload = time.monotonic() + 1800
        while not done.wait(60):
            print(f"[{time.strftime('%H:%M:%S', time.gmtime())} UTC] Current stage: {self.stage}", flush=True)
            if time.monotonic() < max(next_upload, self.last_write + 1800):
                continue
            try:
                self.write_commit(
                    {self.status_key: canonical(dict(stage=self.stage, time=time.time(), owner=self.owner))},
                    "30 minute status",
                )
                next_upload = time.monotonic() + 1800
            except Exception as exc:
                self.problem = exc
                return


@contextmanager
def session(repo, scratch, role="preflight"):
    if (
        not Path("/kaggle/working").is_dir()
        or torch.cuda.device_count() != 2
        or any("T4" not in torch.cuda.get_device_name(i) for i in range(2))
    ):
        raise RuntimeError("Run this notebook on Kaggle with GPU T4 x2.")
    if not repo or repo == "YOUR_ACCOUNT/YOUR_DATASET":
        raise ValueError("Set HF_REPO to your existing writable HF dataset.")
    from kaggle_secrets import UserSecretsClient

    base = Path(scratch).resolve()
    if base not in (Path("/kaggle/temp/kd_view_v2"), Path("/kaggle/working/kd_view_v2")):
        raise ValueError("Choose one of the two documented scratch directories.")
    if role not in ("preflight", "results", *(f"worker-{i}" for i in range(4))):
        raise ValueError("Invalid publisher role")
    base = base / PREFIX.split("/")[-1] / role
    base.mkdir(parents=True, exist_ok=True)
    output = Path("/kaggle/working/kd_v2_receipts") / role
    output.mkdir(parents=True, exist_ok=True)
    h = Publisher(
        repo, UserSecretsClient().get_secret("HF_TOKEN"), output / "transport.jsonl", namespace=PREFIX
    )
    h.lease_key = f"writers/{role}/OWNER.json"
    h.queue_capable = role.startswith("worker-")
    h.status_key = f"writers/{role}/SESSION.json"
    lock = FileLock(str(base / "writer.lock"))
    while True:
        try:
            lock.acquire(timeout=0)
            break
        except Timeout:
            print(f"{role} is already running in this scratch directory. Checking again in 60s.", flush=True)
            time.sleep(60)
    try:
        h.claim()
        done = threading.Event()
        thread = threading.Thread(target=h.pulse, args=(done,), daemon=True)
        thread.start()
        try:
            yield h, base, output
        finally:
            done.set()
            thread.join()
            failure = sys.exc_info()[0]
            status = dict(
                stage=h.stage,
                finished=time.time(),
                writer=h.owner,
                exception=failure.__name__ if failure else None,
            )
            receipt = output / ("session-" + h.owner + ".json")
            json_write(receipt, status)
            if h.problem is None:
                try:
                    with tempfile.TemporaryDirectory(dir=base) as temp:
                        snapshot = Path(temp) / "transport.jsonl"
                        shutil.copyfile(h.log, snapshot)
                        prefix = "sessions/" + h.owner + "/"
                        h.write_commit(
                            {
                                prefix + "transport.jsonl": str(snapshot),
                                prefix + "status.json": canonical(status),
                            },
                            "archive session receipts",
                        )
                except Exception:
                    if failure is None:
                        raise
                finally:
                    h.release()
            elif failure is None:
                raise RuntimeError("Session heartbeat failed. Check local transport.jsonl.") from h.problem
    finally:
        lock.release()


def fetch_json(h, name, file, base, tag=None):
    m, rev = h.selected(name)
    if not m or not m["complete"]:
        return None
    if tag and tag != m["identity"]:
        raise RuntimeError("Different existing identity: " + name)
    with tempfile.TemporaryDirectory(dir=base) as tmp:
        h.fetch(name, m, rev, tmp, [file])
        return json_read(Path(tmp) / file)


def publish_files(h, name, folder, identity, names):
    ck = Checkpoints(folder, identity)
    if not ck.pointer:
        ck.commit(dict(update=1, complete=True), names)
    old, rev = h.selected(name)
    if old and old["identity"] != ck.tag:
        raise RuntimeError("Identity conflict: " + name)
    if not old or not old["complete"]:
        rev = h.publish(name, ck)
    return rev


def load_input(dataset, paths, versions, base, release=False):
    folder = base / "inputs" / dataset
    data, test, clean = load_dataset(
        dataset, mount(dataset, paths.get(dataset)), folder, versions.get(dataset), release
    )
    return data, test, clean, folder


def runtime_id():
    return {**environment(), "timm": md.version("timm"), "pillow": md.version("pillow")}


def check_settings(repo, paths, versions):
    if not repo or repo == "YOUR_ACCOUNT/YOUR_DATASET":
        raise ValueError("Enter your existing HF dataset owner/name in HF_REPO.")
    for ds in SOURCES:
        print(ds, mount(ds, paths.get(ds)), "pinning the input file hashes", flush=True)


def previous_cifar_forecast(h, base, dataset, identity, old_source, model_checks):
    if not old_source or dataset not in ("cifar10", "cifar100"):
        return None
    old = copy(h)
    old.namespace = PREVIOUS_PREFIX
    name = "qualification-" + dataset
    expected = {**identity, "source": old_source}
    manifest, revision = old.selected(name)
    if not manifest or not manifest["complete"] or manifest["identity"] != hash_object(expected):
        return None
    with tempfile.TemporaryDirectory(dir=base) as tmp:
        old.fetch(name, manifest, revision, tmp, ["qualification.json"])
        record = json_read(Path(tmp) / "qualification.json")
    if record["identity"] != expected or record["checked_model_parameters"] != model_checks:
        return None
    return record["forecasts"], {"namespace": PREVIOUS_PREFIX, "revision": revision, "identity": expected}


def qualify(h, base, out, paths, versions, previous_cifar_source=None):
    source = signature()
    model_checks = {}
    with torch.inference_mode():
        for classes, size in ((10, 32), (100, 32), (200, 64)):
            for architecture in ("resnet18", "resnet34", "mobilenetv4_conv_small"):
                net = model_for(architecture, classes).to(DEV).eval()
                z = net(torch.zeros(2, 3, size, size, device=DEV))
                if tuple(z.shape) != (2, classes) or not torch.isfinite(z).all():
                    raise RuntimeError(f"Invalid {architecture} output on {size} px")
                model_checks[f"{architecture}/{classes}"] = sum(p.numel() for p in net.parameters())
                del net, z
    h.stage = "cross-session recovery anchor"
    anchor = "qualification-anchor-" + source[:16]
    with tempfile.TemporaryDirectory(dir=base) as tmp:
        tmp = Path(tmp)
        for i in range(2):
            qualification(tmp / f"gpu{i}", f"cuda:{i}", all_arms=i == 0)
        child(tmp / "anchor", DEV, until=2)
        _, _, store = unit(tmp / "anchor", "teacher", DEV)
        prior, rev = h.selected(anchor)
        if not prior:
            rev = h.publish(anchor, store)
        print("Remote recovery anchor", rev, flush=True)
    for dataset in SOURCES:
        h.stage = "input and capacity " + dataset
        data, _, _, folder = load_input(dataset, paths, versions, base)
        identity = {"source": source, "input": data.identity, "environment": runtime_id()}
        name = "qualification-" + dataset
        cached = fetch_json(h, name, "qualification.json", base, hash_object(identity))
        if cached:
            print("Verified previous qualification", dataset, flush=True)
            continue
        prior = previous_cifar_forecast(h, base, dataset, identity, previous_cifar_source, model_checks)
        forecasts, reused_from = prior if prior else ({}, None)
        if prior:
            print("Reused verified CIFAR qualification:", dataset, flush=True)
        for model, epochs in (("resnet34", 90), ("resnet18", 60), ("mobilenetv4_conv_small", 60)):
            if model in forecasts:
                continue
            probe_name = f"probe-{dataset}-{model}"
            probe_id = {**identity, "model": model}
            cached_probe = fetch_json(h, probe_name, "forecast.json", base, hash_object(probe_id))
            if cached_probe:
                forecasts[model] = cached_probe
                continue
            h.stage = probe_name
            steps = (len(data.fit) + 127) // 128
            print(f"{probe_name}: starting one epoch, {steps} updates", flush=True)

            def report_probe(store, _):
                h.stage = f"{probe_name}: {store.pointer['update']}/{steps} updates committed locally"

            with tempfile.TemporaryDirectory(dir=base) as tmp:
                forecasts[model] = footprint_probe(data, Path(tmp), model, epochs, DEV, on_commit=report_probe)
            dest = base / probe_name
            json_write(dest / "forecast.json", forecasts[model])
            publish_files(h, probe_name, dest, probe_id, ["forecast.json"])
        json_write(
            folder / "qualification.json",
            dict(
                identity=identity, forecasts=forecasts, anchor=anchor, checked_model_parameters=model_checks,
                reused_from=reused_from,
            ),
        )
        publish_files(
            h,
            name,
            folder,
            identity,
            ["qualification.json", "input.json", "splits.csv", "source_files.csv"],
        )
        shutil.copyfile(folder / "qualification.json", out / (dataset + "-qualification.json"))
        print("Qualified", dataset, flush=True)
    return "Qualification complete. Start training in a fresh Kaggle session."


def restore_gate(h, base, anchor):
    manifest, rev = h.selected(anchor)
    if not manifest:
        raise RuntimeError("Run notebook 01 first.")
    with tempfile.TemporaryDirectory(dir=base) as tmp:
        tmp = Path(tmp)
        # Immutable, digest-checked anchor files can be reused on a same-disk restart.
        cached = base / "recovery_anchors" / anchor / manifest["chain"]
        h.fetch(anchor, manifest, rev, cached)
        shutil.copytree(cached, tmp / "restored")
        child(tmp / "restored", DEV)
        child(tmp / "reference", DEV)
        check = verify_continuation(tmp / "reference", tmp / "restored", DEV)
    return {"revision": rev, **check}


def active_rows(group):
    rows = matrix()
    if group == "all":
        return rows
    if group == "pilot":
        return [
            r
            for r in rows
            if r["dataset"] == "cifar100"
            and r["model"] in ("resnet34", "resnet18")
            and r["seed"] in ("907", "42")
        ]
    if group in SOURCES:
        return [r for r in rows if r["dataset"] == group]
    raise ValueError("GROUP options: pilot, all, cifar100, cifar10, tiny_imagenet200")


<<<<<<< HEAD
def plan_session(h, rows, teachers, worker_id, source, models, limit):
    if type(limit) is not int or limit < 0 or not isinstance(models, (list, tuple)):
        raise ValueError("Use a nonnegative MAX_RUNS and a list of MODEL_FILTER names.")
    if set(models) - {r["model"] for r in rows}:
        raise ValueError("MODEL_FILTER contains a model outside this worker's assignments.")
=======
def migrated_inventory(h, rows, teachers, source):
>>>>>>> 7c4a96c (updated again)
    revision = h.head()
    ids = {r["run_id"] for r in rows} | {r["run_id"] for r in teachers.values()}
    paths = [f"{h.namespace}/{name}/MANIFEST.json" for run in sorted(ids) for name in (run, run+"-review")]
    present = set()
    for start in range(0, len(paths), 50):
        batch = paths[start:start+50]
        present.update(item.path for item in h.call(lambda: h.api.get_paths_info(
            h.repo, batch, repo_type="dataset", revision=revision)))
    audited, resumed = set(), set()
    for run in ids:
        if f"{h.namespace}/{run}/MANIFEST.json" not in present:
            continue
        resumed.add(run)
        if f"{h.namespace}/{run}-review/MANIFEST.json" not in present:
            continue
        fit = h.read(run+"/MANIFEST.json", revision)
        review = h.read(run+"-review/MANIFEST.json", revision)
        if not (fit["complete"] and review["complete"] and review["identity"] == hash_object(
                {"fit": fit["identity"], "chain": fit["chain"], "review_source": source})):
            raise RuntimeError("Inconsistent published audit for " + run)
        audited.add(run)
<<<<<<< HEAD
    mine = {r["run_id"] for r in rows}
    h.audited_runs = frozenset(mine & audited)
    h.resumable_runs = frozenset((resumed - audited) & mine)
    pending = [r for r in rows if r["run_id"] not in audited and (not models or r["model"] in models)]
    needed = {r["teacher_id"] for r in pending if r["teacher_id"]}
    pending += [r for r in rows if r["run_id"] in needed - audited and r not in pending]
    def priority(r):
        dep = r["teacher_id"]
        return (bool(dep and dep not in audited), r["role"] != "teacher",
                r["run_id"] not in resumed, int(r["sequence"]))
    pending.sort(key=priority)
    selected = pending[:limit] if limit else pending
    h.session_runs = tuple(r["run_id"] for r in selected)
    print(f"Worker {worker_id}: {len(rows)} assigned, {len(h.audited_runs)} audited, "
          f"{len(selected)} selected for this session.", flush=True)
    for r in selected:
        print(" ", r["run_id"], r["epochs"], "epochs", r["updates"], "updates", flush=True)
    return selected, audited


def train(h, base, out, paths, versions, group, num_workers, worker_id, models=(), limit=0, hours=0):
    if group != "all" or type(worker_id) is not int or not 0 <= worker_id < num_workers:
        raise ValueError("Use GROUP='all' and a worker ID from 0 through 3.")
    full = active_rows("all")
    owners = worker_plan(full, num_workers)
    rows = [r for r in full if owners[r["run_id"]] == worker_id]
    teachers = {r["dataset"]: r for r in full if r["role"] == "teacher"}
    print(f"Worker {worker_id} owns {len(rows)} of {len(full)} fits.", flush=True)
    source = signature()
    if not isinstance(hours, (float, int)) or not 0 <= hours <= 168:
        raise ValueError("SESSION_HOURS must be between 0 and 168 (0 disables the time limit).")
    deadline = time.monotonic() + hours * 3600 if hours else float("inf")
    h.stage = "checking published audits"
    rows, audited = plan_session(h, rows, teachers, worker_id, source, models, limit)
    if not rows:
        return "No pending fits match this session."
    # Process ready datasets before batches that depend on another worker's teacher.
    batches = {}
    for r in rows:
        dep = r["teacher_id"]
        blocked = bool(dep and dep not in audited and owners[dep] != worker_id)
        batches.setdefault((blocked, r["dataset"]), []).append(r)
    order = sorted(batches, key=lambda item: item[0])
    first_dataset = order[0][1]
    h.stage = "checking recovery and preparing " + first_dataset
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending_input = pool.submit(load_input, first_dataset, paths, versions, base)
        gate = restore_gate(h, base, "qualification-anchor-" + source[:16])
        json_write(out / "cross_session_restore.json", gate)
        h.write_commit({"sessions/"+h.owner+"/restore.json": canonical(gate)}, "verified recovery receipt")
        first_input = pending_input.result()
    del pending_input
    for index, (blocked, dataset) in enumerate(order):
        if time.monotonic() >= deadline:
            h.session_paused = True
            return "Session time budget reached."
        h.stage = "preparing " + dataset
        if index == 0:
            data = first_input[0]
            del first_input
        else:
            data = load_input(dataset, paths, versions, base)[0]
        identity = {"source": source, "input": data.identity, "environment": runtime_id()}
        check = fetch_json(h, "qualification-" + dataset, "qualification.json", base, hash_object(identity))
        if not check:
            raise RuntimeError("Missing matching qualification for " + dataset)
        print("Verified preflight source, environment and input for", dataset, flush=True)
        d = Driver(
            base / dataset,
            h,
            data,
            DEV,
            {},
            {"protocol_sha256": file_info(HERE / "KAGGLE_EXPERIMENT_PLAN.md")["sha256"]},
        )
        d.source = source
        d.deadline = deadline
=======
    return audited, resumed - audited


def read_teacher(h, base, d, teacher_run, source):
    completed, _ = h.selected(teacher_run + "-review")
    if not completed or not completed["complete"]:
        raise RuntimeError("Missing verified teacher audit for " + teacher_run)
    manifest, rev = h.selected(teacher_run)
    with tempfile.TemporaryDirectory(dir=base) as tmp:
        tmp = Path(tmp)
        h.fetch(teacher_run, manifest, rev, tmp, ["config.json", "ENDPOINT.json"])
        contract = json_read(tmp / "config.json")["contract"]
        if (
            not manifest["complete"]
            or contract["source"] != source
            or contract["data"] != d.data.identity
            or contract["protocol_sha256"] != d.receipt["protocol_sha256"]
            or completed["identity"]
            != hash_object(
                {
                    "fit": manifest["identity"],
                    "chain": manifest["chain"],
                    "review_source": source,
                }
            )
        ):
            raise ValueError("Shared teacher belongs to a different experiment")
        best = json_read(tmp / "ENDPOINT.json")["best"]["weights"]
        h.fetch(teacher_run, manifest, rev, tmp, [best])
        teacher = model_for("resnet34", d.data.classes).cpu().eval().requires_grad_(False)
        teacher.load_state_dict(
            torch.load(tmp / best, map_location="cpu", weights_only=True)
        )
        tid = manifest["files"][best]["sha256"]
    return teacher, tid
>>>>>>> 7c4a96c (updated again)


def train(h, base, out, paths, versions, group, num_workers, worker_id, models=(), limit=0, hours=0):
    from kaggle_dispatch import Dispatch

    if group != "all" or num_workers != 4 or worker_id not in range(4):
        raise ValueError("Use GROUP='all', four workers and a worker ID from 0 through 3.")
    if h.repo != "Shanmuk4622/kd-crop-budget-4w-2026":
        raise ValueError("This shared queue is bound to the campaign's configured HF repo.")
    full = matrix()
    if type(limit) is not int or limit < 0 or not isinstance(models, (tuple, list)) or (
        set(models) - {r["model"] for r in full}
    ):
        raise ValueError("Invalid MAX_RUNS or MODEL_FILTER")
    if not isinstance(hours, (float, int)) or not 0 <= hours <= 168:
        raise ValueError("SESSION_HOURS must be between 0 and 168.")
    deadline = time.monotonic() + hours * 3600 if hours else float("inf")
    owners = worker_plan(full, 4)
    original = [r for r in full if owners[r["run_id"]] == worker_id]
    teachers = {r["dataset"]: r for r in full if r["role"] == "teacher"}
    source = signature()
    queue = Dispatch(worker_id, h.owner)
    h.queue = queue
    migrate = queue.call("ping")["migration_required"]
    if migrate or not getattr(h, "queue_role", False):
        h.stage = "verifying prior Hugging Face audits for queue migration"
        audited, partial = migrated_inventory(h, original, teachers, source)
        estimates = {}
        for dataset in sorted({r["dataset"] for r in original}):
            proof = fetch_json(h, "qualification-" + dataset, "qualification.json", base)
            if proof and proof["identity"]["source"] == source:
                for r in original:
                    if r["dataset"] == dataset:
                        estimates[r["run_id"]] = int(r["updates"]) * proof["forecasts"][r["model"]]["measured_step_seconds"]
        with h.guard:
            h.queue_role = True
            h.write_commit({}, "retire fixed assignments for this worker")
        queue.call("migrate", dict(audited=sorted(audited), resumable=sorted(partial), estimates=estimates))
    by_id = {r["run_id"]: r for r in full}
    prepared = {}
    recovery_ok = False
    finished = 0
    while not limit or finished < limit:
        if time.monotonic() >= deadline:
            h.session_paused = True
            return "Session time budget reached."
        h.stage = "requesting a ready fit from the shared queue"
        job = queue.call("claim", {"models": list(models)})
        if job.get("finished"):
            return "All matching fits are audited."
        if job.get("waiting"):
            h.stage = "waiting for a ready fit: see teacher, bank and migration status in the queue"
            teacher_run = job.get("check_teacher")
            if teacher_run:
                verified, _ = migrated_inventory(h, [by_id[teacher_run]], {}, source)
                if teacher_run in verified:
                    queue.call("observe_teacher", {"run": teacher_run})
                    continue
            time.sleep(min(30, max(0, deadline-time.monotonic())))
            continue
        r = by_id[job["run"]]
        dataset = r["dataset"]
        try:
<<<<<<< HEAD
            teacher, tid = None, None
            local_rows = batches[blocked, dataset]
            # The fourth worker can fit the independent hard-label controls immediately.
            if worker_id == 3:
                local_rows.sort(key=lambda r: r["arm"] != "hard_labels")
            for r in local_rows:
                h.stage = r["run_id"]
                h.fit_started_at = time.time()
                print(h.stage, flush=True)
                arm = "teacher" if r["role"] == "teacher" else r["arm"]
                cfg = dict(
                    model=r["model"],
                    arm=arm,
                    seed=int(r["seed"]),
                    epochs=int(r["epochs"]),
                    batch=128,
                    warm_epochs=5,
                    commit_updates=32,
                )
                budget = check["forecasts"][r["model"]]["required_incremental_bytes"]
                if arm == "teacher":
                    teacher, tid = d.fit(cfg, budget)
                else:
                    if arm != "hard_labels" and teacher is None:
                        tr = teachers[dataset]
                        owner = owners[tr["run_id"]]
                        print(f"Awaiting verified {dataset} teacher from worker {owner}", flush=True)
                        h.stage = f"waiting for {dataset} teacher (worker {owner})"
                        wait_seconds = 30
                        while True:
                            if time.monotonic() >= deadline:
                                h.session_paused = True
                                return "Session time budget reached while waiting for a teacher."
                            if d.stopping:
                                raise KeyboardInterrupt
                            if getattr(h, "problem", None):
                                raise RuntimeError("HF heartbeat failed")
                            completed, _ = h.selected(tr["run_id"] + "-review")
                            if completed and completed["complete"]:
                                break
                            time.sleep(min(wait_seconds, max(0, deadline-time.monotonic())))
                            wait_seconds = min(300, wait_seconds*2)
                        # A completed, audited teacher has immutable selected weights.
                        # Read the endpoint directly; only the owner runs the teacher writer.
                        manifest, rev = h.selected(tr["run_id"])
                        with tempfile.TemporaryDirectory(dir=base) as tmp:
                            tmp = Path(tmp)
                            h.fetch(tr["run_id"], manifest, rev, tmp, ["config.json", "ENDPOINT.json"])
                            contract = json_read(tmp / "config.json")["contract"]
                            if (
                                not manifest["complete"]
                                or contract["source"] != source
                                or contract["data"] != data.identity
                                or contract["protocol_sha256"] != d.receipt["protocol_sha256"]
                                or completed["identity"]
                                != hash_object(
                                    {
                                        "fit": manifest["identity"],
                                        "chain": manifest["chain"],
                                        "review_source": source,
                                    }
                                )
                            ):
                                raise ValueError("Shared teacher belongs to a different experiment")
                            best = json_read(tmp / "ENDPOINT.json")["best"]["weights"]
                            h.fetch(tr["run_id"], manifest, rev, tmp, [best])
                            teacher = model_for("resnet34", data.classes).cpu().eval().requires_grad_(False)
                            teacher.load_state_dict(
                                torch.load(tmp / best, map_location="cpu", weights_only=True)
                            )
                            tid = manifest["files"][best]["sha256"]
                        h.stage = r["run_id"]
                    bank = bs = manifest = None
                    if arm not in ("live_kd", "hard_labels"):
                        bank, bs, manifest = d.bank(cfg["seed"], arm, teacher, tid)
                    d.fit(
                        cfg,
                        budget,
                        teacher if arm == "live_kd" else None,
                        tid if arm != "hard_labels" else None,
                        bank,
                    )
                    if bs:
                        del bank
                        erase_verified_copies(bs, manifest)
                json_write(
                    out / (dataset + "-latest.json"),
                    {"finished": r["run_id"], "revision": d.archives[r["run_id"]]["revision"]},
                )
                h.audited_runs = h.audited_runs | {r["run_id"]}
                if d.stopping:
                    raise KeyboardInterrupt
        except KeyboardInterrupt:
            if time.monotonic() < deadline:
                raise
            h.session_paused = True
            return "Session time budget reached at a checkpoint."
        finally:
            for s, handler in prev.items():
                signal.signal(s, handler)
    return f"Completed requested group {group}."
=======
            with queue.holding(h, job):
                verified = migrated_inventory(h, [r], {}, source)[0] if job.get("resumable") else set()
                if job["run"] in verified:
                    print("Recovered verified audit for", job["run"], flush=True)
                else:
                    if dataset not in prepared:
                        h.stage = "loading and checking " + dataset
                        with ThreadPoolExecutor(max_workers=1) as pool:
                            loaded = pool.submit(load_input, dataset, paths, versions, base)
                            if not recovery_ok:
                                gate = restore_gate(h, base, "qualification-anchor-" + source[:16])
                                json_write(out / "cross_session_restore.json", gate)
                                h.write_commit({"sessions/"+h.owner+"/restore.json": canonical(gate)}, "verified recovery receipt")
                                recovery_ok = True
                            data = loaded.result()[0]
                        identity = {"source": source, "input": data.identity, "environment": runtime_id()}
                        check = fetch_json(h, "qualification-" + dataset, "qualification.json", base, hash_object(identity))
                        if not check:
                            raise RuntimeError("Missing matching qualification for " + dataset)
                        print("Verified preflight source, environment and input for", dataset, flush=True)
                        d = Driver(base / dataset, h, data, DEV, {},
                                   {"protocol_sha256": file_info(HERE / "KAGGLE_EXPERIMENT_PLAN.md")["sha256"]})
                        d.source, d.deadline = source, deadline
                        prepared[dataset] = d, check
                    d, check = prepared[dataset]
                    # Each owner starts from the selected HF chain, even when a stale local
                    # directory survives a transfer to another notebook and back.
                    d.base = base / "claims" / job["ticket"] / dataset
                    h.fit_root = d.base
                    execute_claim(h, base, out, d, check, r, queue, job)
            queue.call("finish", job)
            finished += 1
        except KeyboardInterrupt:
            # Driver publishes a recoverable checkpoint on a catchable training interruption.
            manifest, _ = h.selected(r["run_id"])
            queue.call("yield", {**job, "resumable": bool(manifest)})
            if time.monotonic() >= deadline:
                h.session_paused = True
                return "Session time budget reached at a checkpoint."
            raise
    return f"Completed {finished} claimed fits."


def execute_claim(h, base, out, d, check, r, queue, job):
    run = r["run_id"]
    arm = "teacher" if r["role"] == "teacher" else r["arm"]
    cfg = dict(model=r["model"], arm=arm, seed=int(r["seed"]), epochs=int(r["epochs"]),
               batch=128, warm_epochs=5, commit_updates=32)
    def stop(*_):
        d.stopping = True
        print("Stopping at the next complete transaction.", flush=True)
    prev = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        bank = store = manifest = teacher = tid = None
        if r["teacher_id"]:
            h.stage = "loading verified teacher for " + run
            if not hasattr(d, "shared_teacher"):
                d.shared_teacher = read_teacher(h, base, d, r["teacher_id"], d.source)
            teacher, tid = d.shared_teacher
        if r["bank_id"]:
            h.stage = "preparing verified prediction bank for " + run
            bank, store, manifest = d.bank(cfg["seed"], arm, teacher, tid)
            queue.call("bank", job)
        h.stage, h.fit_started_at = run, time.time()
        print("Training or resuming", run, flush=True)
        d.fit(cfg, check["forecasts"][r["model"]]["required_incremental_bytes"],
              teacher if arm == "live_kd" else None, tid, bank)
        if store:
            del bank
            erase_verified_copies(store, manifest)
        archive = d.archives[run]
        json_write(out / (r["dataset"] + "-latest.json"),
                   {"finished": run, "revision": archive["revision"], "audit": archive["review_revision"]})
        if d.stopping:
            raise KeyboardInterrupt
    finally:
        for sig, handler in prev.items():
            signal.signal(sig, handler)
>>>>>>> 7c4a96c (updated again)
