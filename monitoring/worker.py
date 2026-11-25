"""Read-only observer for the four Kaggle workers. Embedded in each training notebook."""

import csv
import json
import subprocess
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class FitProgress:
    def __init__(self, base, hub):
        self.base, self.hub = Path(base), hub
        self.last_key, self.metrics = None, {}

    def sample(self):
        run = self.hub.stage
        parts = run.split(".")
        row = dict(run_id=run, model=None, epoch=None, loss=None, accuracy=None,
                   comment=run[:200])
        if len(parts) != 5 or parts[0] != "kaggle-v2":
            return row
        _, dataset, model, arm, seed = parts
        row.update(model=model, comment="Restoring fit or preparing its first batch")
        unit = getattr(self.hub, "fit_root", self.base / dataset) / "units" / run
        pointer = unit / "POINTER.json"
        try:
            st = pointer.stat()
            key = (run, st.st_mtime_ns, st.st_size)
            if key != self.last_key:
                p = json.loads(pointer.read_text())
                step = p["update"]
                metrics = dict(comment="Fit initialized", update=step)
                name = f"updates/{step:08d}/updates.csv"
                if step and name in p["files"]:
                    with (unit / name).open(newline="") as f:
                        batch = list(csv.DictReader(f))[-1]
                    n = int(batch["n"])
                    metrics.update(epoch=int(batch["epoch"]), loss=float(batch["objective_sum"]) / n,
                                   accuracy=100 * int(batch["correct"]) / n,
                                   learning_rate=float(batch["learning_rate"]),
                                   comment=f"Local checkpoint at update {step}")
                    for epoch in (int(batch["epoch"]), int(batch["epoch"]) - 1):
                        filename = f"epochs/{epoch:03d}/metrics.csv"
                        if filename in p["files"]:
                            with (unit / filename).open(newline="") as f:
                                measured = next(csv.DictReader(f))
                            metrics["validation_epoch"] = epoch
                            metrics["validation_accuracy"] = 100 * int(measured["validation_correct"]) / int(measured["validation_n"])
                            break
                if p["complete"]:
                    metrics["comment"] = "Fit complete; verifying or publishing evidence"
                self.last_key, self.metrics = key, metrics
            row.update(self.metrics)
        except (OSError, ValueError, KeyError, IndexError, ZeroDivisionError):
            # Evidence may already have been archived or a new fit may be starting.
            pass
        return row


def gpu_metrics():
    try:
        query = "index,utilization.gpu,memory.used,memory.total"
        raw = subprocess.check_output(["nvidia-smi", "--query-gpu=" + query,
                                       "--format=csv,noheader,nounits"], text=True, timeout=3,
                                      stderr=subprocess.DEVNULL)
        result = []
        for fields in csv.reader(raw.splitlines(), skipinitialspace=True):
            values = [int(x) if x.strip().isdigit() else None for x in fields]
            index, utilization, used, total = values
            result.append(dict(index=index, utilization=utilization, used_mib=used, total_mib=total))
        return result
    except (OSError, subprocess.SubprocessError, ValueError):
        return []


@contextmanager
def monitor_worker(worker, base, hub, enabled=True):
    if not enabled:
        yield
        return
    try:
        import requests
        import psutil
        from kaggle_secrets import UserSecretsClient
        secret = UserSecretsClient().get_secret
        url = secret("SUPABASE_URL").strip().rstrip("/")
        key = secret("SUPABASE_ANON_KEY").strip()
        token = secret(f"KD_WORKER_TOKEN_{worker}").strip()
        if not url.startswith("https://") or not key or not token:
            raise ValueError("Missing monitor settings")
    except Exception:
        print(f"Monitor offline: check Kaggle secrets for worker {worker} and psutil. Training continues.", flush=True)
        yield
        return

    started = utc_now()
    progress = FitProgress(base, hub)
    stop = threading.Event()
    terminal = ["completed", "Worker finished"]

    def send_loop():
        session = requests.Session()
        seq = 0
        failures = 0
        legacy_until = 0
        psutil.cpu_percent()
        try:
            while True:
                tick = time.monotonic()
                final = stop.is_set()
                try:
                    sample = progress.sample()
                    details = {k: sample.pop(k) for k in ("update","learning_rate","validation_epoch","validation_accuracy") if k in sample}
                    if hasattr(hub, "session_runs"):
                        details.update(audited=sorted(hub.audited_runs), resumable=sorted(hub.resumable_runs),
                                       selected=list(hub.session_runs))
                    if sample["model"] and hasattr(hub, "fit_started_at"):
                        details["fit_started_at"] = hub.fit_started_at
                    cpu = psutil.cpu_percent()
                    status = terminal[0] if final else "running"
                    if final:
                        sample["comment"] = terminal[1]
                    sample.update(worker_id=worker, started_at=started, seq=seq, status=status,
                                  cpu=cpu if seq else None, gpus=gpu_metrics(), sampled_at=utc_now())
                    seq += 1
                    def report(data):
                        return session.post(url + "/rest/v1/rpc/kd_report", headers={"apikey": key},
                                            json={"p_token": token, "p": data}, timeout=(3, 5))
                    extended = tick >= legacy_until
<<<<<<< HEAD
                    response = report({**sample, "details": details} if extended else sample)
                    if extended and response.status_code == 400 and "Invalid worker sample" in response.text:
                        legacy_until = tick + 600
                        response = report(sample)
                    response.raise_for_status()
=======
                    queue = getattr(hub, "queue", None)
                    if not (queue and queue.renew({**sample, "details": details})):
                        response = report({**sample, "details": details} if extended else sample)
                        if extended and response.status_code == 400 and "Invalid worker sample" in response.text:
                            legacy_until = tick + 600
                            response = report(sample)
                        response.raise_for_status()
>>>>>>> 7c4a96c (updated again)
                    if failures or seq == 1:
                        print(f"Supabase monitor connected for worker {worker}.", flush=True)
                    failures = 0
                except Exception as exc:
                    failures += 1
                    if failures == 1 or failures % 10 == 0:
                        code = getattr(getattr(exc, "response", None), "status_code", None)
                        print(f"Monitor sample unavailable ({code or type(exc).__name__}); training continues.", flush=True)
                if final:
                    return
                stop.wait(max(0, 30 - (time.monotonic() - tick)))
        finally:
            session.close()

    thread = threading.Thread(target=send_loop, name="kd-monitor", daemon=True)
    thread.start()
    try:
        yield
        if getattr(hub, "session_paused", False):
            terminal[:] = ["stopped", "Session budget reached"]
        elif hasattr(hub, "session_runs"):
            terminal[1] = "Selected session work finished"
    except KeyboardInterrupt:
        terminal[:] = ["stopped", "Worker interrupted"]
        raise
    except BaseException as exc:
        terminal[:] = ["failed", "Worker stopped: " + type(exc).__name__]
        raise
    finally:
        stop.set()
        thread.join(timeout=24)
