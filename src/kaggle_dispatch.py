"""Supabase coordination for one frozen Kaggle campaign. HF remains the checkpoint store."""
import threading
import time
from contextlib import contextmanager

import requests


class Dispatch:
    def __init__(self, worker, session_id):
        from kaggle_secrets import UserSecretsClient

        secret = UserSecretsClient().get_secret
        self.url = secret("SUPABASE_URL").strip().rstrip("/") + "/rest/v1/rpc/kd_dispatch"
        self.key = secret("SUPABASE_ANON_KEY").strip()
        self.auth = dict(p_worker=worker, p_token=secret(f"KD_WORKER_TOKEN_{worker}").strip(),
                         p_session=session_id)
        self.job = None
        self.renewed = 0
        self.lost = None

    def call(self, action, data=None):
        for attempt in range(4):
            try:
                response = requests.post(self.url, headers={"apikey": self.key},
                    json={**self.auth, "p_action": action, "p_data": data or {}}, timeout=(5, 20))
                if response.ok:
                    return response.json()
                if response.status_code == 404:
                    raise RuntimeError("Install monitoring/queue.sql in the Supabase SQL Editor.")
                if response.status_code not in (429, 500, 502, 503, 504) or attempt == 3:
                    message = response.json().get("message", "queue request rejected")
                    raise RuntimeError(f"Queue {action}: {message}")
            except (requests.Timeout, requests.ConnectionError):
                if attempt == 3:
                    raise
            time.sleep(2 ** attempt)

    def renew(self, sample=None):
        job = self.job
        if job is None:
            return False
        if self.lost:
            raise self.lost
        data = {**job, "sample": sample} if sample is not None else job
        self.call("renew", data)
        if self.job is job:
            self.renewed = time.monotonic()
        return True

    @contextmanager
    def holding(self, hub, job):
        self.job = job
        self.lost = None
        self.renewed = time.monotonic()
        done = threading.Event()

        def backup():
            while not done.wait(20):
                if time.monotonic() - self.renewed < 45:
                    continue
                try:
                    self.renew()
                except Exception as exc:
                    if isinstance(exc, RuntimeError) or time.monotonic() - self.renewed >= 240:
                        self.lost = RuntimeError(f"Queue heartbeat failed: {exc}")
                        return

        thread = threading.Thread(target=backup, daemon=True)
        thread.start()
        try:
            with hub.guard:
                hub.before_commit = self.renew
                # The new HF parent fences delayed commits from a previous claim.
                hub.write_commit({}, "claim fit " + job["run"])
            yield
        finally:
            done.set()
            thread.join()
            with hub.guard:
                hub.before_commit = None
            self.job = None
