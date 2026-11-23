"""Per-board readings over synchronized windows. Missing counters stay missing."""

import math
import threading
import time
import uuid
from pathlib import Path

import torch

from kd_fit_io import csv_write


def energy_from_rows(rows, duration):
    if not rows or duration <= 0:
        return None, "unavailable", 0.0
    t0, t1 = rows[0][0], rows[-1][0]
    covered = max(0, t1 - t0) / duration
    counters = [r[2] for r in rows if r[2] is not None]
    if (
        covered >= 0.98
        and duration - (t1 - t0) <= 0.5
        and rows[0][2] is not None
        and rows[-1][2] is not None
        and rows[-1][2] > rows[0][2]
        and all(a <= b for a, b in zip(counters, counters[1:]))
    ):
        return (rows[-1][2] - rows[0][2]) / 1000, "nvml_counter_mJ", covered
    valid = [r for r in rows if r[1] is not None]
    if len(valid) < 2:
        return None, "no_power_samples", 0.0
    coverage = (valid[-1][0] - valid[0][0]) / duration
    intervals = [b[0] - a[0] for a, b in zip(valid, valid[1:])]
    if coverage < 0.98 or max(intervals) > 1 or valid[0][0] - t0 > 0.25 or t1 - valid[-1][0] > 0.25:
        return None, "insufficient_coverage", coverage
    joules = math.fsum((a[1] + b[1]) * 0.5 * (b[0] - a[0]) for a, b in zip(valid, valid[1:]))
    return joules, "trapezoidal_power_W", coverage


class BoardWindow:
    def __init__(self, folder, phase, device):
        self.folder, self.phase, self.device = Path(folder), phase, device
        self.folder.mkdir(parents=True, exist_ok=True)
        self.nv = None
        self.handles = []
        self.error = ""
        self.active_uuid = ""
        if device.type == "cuda":
            properties = torch.cuda.get_device_properties(device)
            self.active_uuid = str(getattr(properties, "uuid", "")).lower().removeprefix("gpu-")
        try:
            import pynvml as nv

            nv.nvmlInit()
            self.nv = nv
            for i in range(nv.nvmlDeviceGetCount()):
                handle = nv.nvmlDeviceGetHandleByIndex(i)
                self.handles.append((str(nv.nvmlDeviceGetUUID(handle)), handle))
        except Exception as exc:
            self.error = type(exc).__name__
        self.rows = [[] for _ in self.handles]
        self.stopped = threading.Event()

    def sample(self):
        for i, (_, handle) in enumerate(self.handles):
            now = time.monotonic()
            try:
                watts = self.nv.nvmlDeviceGetPowerUsage(handle) / 1000
            except Exception:
                watts = None
            try:
                energy = self.nv.nvmlDeviceGetTotalEnergyConsumption(handle)
            except Exception:
                energy = None
            self.rows[i].append((now, watts, energy))

    def start(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.begin = time.monotonic()
        self.sample()

        def monitor():
            while not self.stopped.wait(0.2):
                self.sample()

        self.thread = threading.Thread(target=monitor, daemon=True)
        self.thread.start()
        return self

    def finish(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.stopped.set()
        self.thread.join()
        self.sample()
        elapsed = time.monotonic() - self.begin
        key = uuid.uuid4().hex
        raw = self.folder / (key + "_samples.csv")
        summary = self.folder / (key + "_phase.csv")
        csv_write(
            raw,
            ["window", "phase", "nvml_index", "uuid", "monotonic_s", "power_W", "cumulative_mJ"],
            (
                (key, self.phase, i, self.handles[i][0], t, p, e)
                for i, rows in enumerate(self.rows)
                for t, p, e in rows
            ),
        )
        results = []
        for i, rows in enumerate(self.rows):
            joules, method, coverage = energy_from_rows(rows, elapsed)
            unit_uuid = self.handles[i][0].lower().removeprefix("gpu-")
            active = int(unit_uuid == self.active_uuid) if self.active_uuid else ""
            results.append(
                (
                    key,
                    self.phase,
                    i,
                    self.handles[i][0],
                    active,
                    elapsed,
                    joules,
                    method,
                    coverage,
                    len(rows),
                )
            )
        if not results:
            results.append(
                (key, self.phase, "", "", "", elapsed, "", self.error or "no_NVML_devices", 0, 0)
            )
        csv_write(
            summary,
            [
                "window",
                "phase",
                "nvml_index",
                "uuid",
                "active_board",
                "wall_s",
                "board_J",
                "method",
                "coverage",
                "observations",
            ],
            results,
        )
        if self.nv is not None:
            try:
                self.nv.nvmlShutdown()
            except Exception:
                pass
        return raw, summary
