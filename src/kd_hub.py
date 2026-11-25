"""One-owner, revision-pinned publication of direct CSV and native training evidence."""

import hashlib
import random
import re
import time
import uuid
from pathlib import Path

from kd_fit_io import canonical, file_info, json_read, safe_path, write_atomic


def retry_delay(exc, attempt):
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status not in (429, 500, 502, 503, 504) and (
        status is not None
        or not any(s in type(exc).__name__.lower() for s in ("timeout", "connection", "transport"))
    ):
        return None
    headers = getattr(response, "headers", {}) or {}
    value = headers.get("Retry-After", "0")
    try:
        wait = float(value)
    except ValueError:
        from email.utils import parsedate_to_datetime

        wait = max(0, parsedate_to_datetime(value).timestamp() - time.time())
    reset = re.search(r"\bt=(\d+)", headers.get("RateLimit", ""))
    return (
        max(wait, float(reset.group(1)) if reset else 0, min(45, 2**attempt)) + random.SystemRandom().random()
    )


class WriterLeaseBusy(RuntimeError):
    def __init__(self, key, expires):
        self.expires = expires
        deadline = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(expires))
        super().__init__(f"{key} has an active writer lease until {deadline}")


class Hub:
    def __init__(self, repo, token, log, namespace="kd-view-budget/v1"):
        from huggingface_hub import HfApi
        import requests

        if not token or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
            raise ValueError("A dedicated existing dataset repository and HF_TOKEN are required")
        self.repo, self.namespace, self.log = repo, namespace, Path(log)
        self.api = HfApi(token=token)
        self.http = requests.Session()
        self.http.headers.update(Authorization="Bearer " + token)
        self.owner = uuid.uuid4().hex
        self.lease_key = "OWNER.json"
        self.last_verified = {}
        self._parent = None
        self.last_write = 0.0

    def record(self, op, **fields):
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with self.log.open("ab") as f:
            f.write(canonical(dict(operation=op, unix_time=time.time(), **fields)) + b"\n")

    def call(self, fn):
        for attempt in range(6):
            try:
                return fn()
            except Exception as exc:
                delay = retry_delay(exc, attempt)
                self.record("request_failure", error_type=type(exc).__name__, delay=delay)
                if delay is None or attempt == 5:
                    raise
                deadline = time.monotonic() + delay
                while time.monotonic() < deadline:
                    time.sleep(min(30, max(0, deadline - time.monotonic())))

    def head(self):
        return self.call(lambda: self.api.repo_info(self.repo, repo_type="dataset").sha)

    def read(self, path, rev):
        def fetch():
            with self.http.get(
                f"https://huggingface.co/datasets/{self.repo}/resolve/{rev}/{self.namespace}/{path}",
                timeout=(10, 90),
            ) as response:
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                return response.json()

        return self.call(fetch)

    def write_commit(self, changes, label, claim=False, release=False):
        from huggingface_hub import CommitOperationAdd

        for attempt in range(20):
            # An accepted parent fences every write. Refresh ownership on any conflict.
            if self._parent and not claim:
                head, lease = self._parent, {"owner": self.owner}
            else:
                head = self.head()
                lease = self.read(self.lease_key, head) or {}
<<<<<<< HEAD
=======
                if getattr(self, "queue_capable", False) and "queue_owner" in lease:
                    self.queue_role = True
                    lease = {"owner": lease["queue_owner"], "expires": lease["queue_expires"]}
>>>>>>> 7c4a96c (updated again)
            other = lease.get("owner") not in (None, self.owner)
            if claim and other and lease.get("expires", 0) > time.time():
                raise WriterLeaseBusy(self.lease_key, lease["expires"])
            if not claim and lease.get("owner") != self.owner:
                raise RuntimeError(f"Writer ownership changed at {self.lease_key}. Stopping publication.")
            lease = {
                "owner": None if release else self.owner,
                "expires": 0 if release else time.time() + 2700,
            }
            if getattr(self, "queue_role", False):
                # Old notebooks read only owner/expires and must never reopen migrated fits.
                lease = {"owner": "shared-queue-v1", "expires": 4102444800,
                         "queue_owner": lease["owner"], "queue_expires": lease["expires"]}
            values = {**changes, self.lease_key: canonical(lease)}
            ops = [
                CommitOperationAdd(path_in_repo=self.namespace + "/" + n, path_or_fileobj=v)
                for n, v in values.items()
            ]
            # Check the job claim after choosing this specific CAS parent, including on retries.
            before_commit = getattr(self, "before_commit", None)
            if before_commit:
                before_commit()
            try:
                rev = self.api.create_commit(
                    self.repo,
                    repo_type="dataset",
                    parent_commit=head,
                    operations=ops,
                    commit_message="KD view study " + label,
                ).oid
                self.record("commit", revision=rev, files=len(ops), label=label)
                self._parent = None if release else rev
                self.last_write = time.monotonic()
                return rev
            except Exception as exc:
                self._parent = None
                code = getattr(getattr(exc, "response", None), "status_code", None)
                if code in (409, 412) and attempt < 19:
                    time.sleep(random.SystemRandom().uniform(0.2, min(5, 0.5 + attempt)))
                    continue
                # A timeout may follow successful selection or release. Recognize its marker.
                current = self.head()
                if self.read(self.lease_key, current) == lease and all(
                    self.read(n, current) == __import__("json").loads(v)
                    for n, v in changes.items()
                    if isinstance(v, bytes) and n.endswith(".json")
                ):
                    self._parent = None if release else current
                    self.last_write = time.monotonic()
                    return current
                wait = retry_delay(exc, attempt)
                if wait is None or attempt >= 5:
                    raise
                time.sleep(wait)
        raise RuntimeError("Could not acquire a consistent Hub parent revision")

    def claim(self):
        self.write_commit({}, "claim writer", claim=True)

    def release(self):
        self.write_commit({}, "release writer", release=True)

    def selected(self, unit):
        revision = self.head()
        return self.read(unit + "/MANIFEST.json", revision), revision

    def publish(self, unit, store, extras=None):
        local = {**store.snapshot_files(), **(extras or {})}
        if not local:
            return None
        old, revision = self.selected(unit)
        if old and old["identity"] != store.tag:
            raise RuntimeError("Remote scientific identity differs")
        entries = {}
        for name, info in local.items():
            path = Path(name)
            remote = (path.parent / (path.stem + "-" + info["sha256"][:24] + path.suffix)).as_posix()
            entries[name] = {**info, "object": unit + "/objects/" + remote}
        changed = [n for n, v in entries.items() if not old or old["files"].get(n) != v]
        if (not changed and old and old["files"] == entries
                and old["chain"] == store.pointer["chain"] and old["complete"] == store.pointer["complete"]):
            self.last_verified[unit] = revision
            return revision
        for start in range(0, len(changed), 128):
            names = changed[start : start + 128]
            changes = {}
            for name in names:
                source = store.path(name)
                if file_info(source) != local[name]:
                    raise RuntimeError("File changed during snapshot: " + name)
                changes[entries[name]["object"]] = str(source)
            revision = self.write_commit(changes, "stage evidence")
        for start in range(0, len(changed), 48):
            names = changed[start : start + 48]
            paths = [self.namespace + "/" + entries[n]["object"] for n in names]
            infos = self.call(
                lambda: self.api.get_paths_info(self.repo, paths, repo_type="dataset", revision=revision)
            )
            by_name = {row.path: row for row in infos}
            for name, path in zip(names, paths):
                info = by_name[path]
                if info.size != local[name]["bytes"]:
                    raise RuntimeError("Remote byte count mismatch: " + name)
                lfs = getattr(info, "lfs", None)
                if lfs:
                    actual = lfs["sha256"] if isinstance(lfs, dict) else lfs.sha256
                    if actual != local[name]["sha256"]:
                        raise RuntimeError("Remote LFS digest mismatch")
                else:
                    body = store.path(name).read_bytes()
                    git_id = hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()
                    if info.blob_id != git_id:
                        raise RuntimeError("Remote Git blob mismatch")
        snapshot = {
            "identity": store.tag,
            "generation": store.pointer["generation"],
            "chain": store.pointer["chain"],
            "complete": store.pointer["complete"],
            "files": entries,
        }
        rev = self.write_commit({unit + "/MANIFEST.json": canonical(snapshot)}, "select " + unit)
        if self.read(unit + "/MANIFEST.json", rev) != snapshot:
            raise RuntimeError("Selecting manifest failed verification")
        self.last_verified[unit] = rev
        self.record("verified_manifest", unit=unit, revision=rev, generation=snapshot["generation"])
        return rev

    def fetch(self, unit, manifest, rev, folder, names=None):
        for name in (manifest["files"] if names is None else names):
            meta = manifest["files"][name]
            target = safe_path(folder, name)
            if target.exists() and file_info(target) == {k: meta[k] for k in ("bytes", "sha256")}:
                continue
            obj = meta["object"]
            if not obj.startswith(unit + "/objects/") or ".." in Path(obj).parts:
                raise RuntimeError("Invalid remote object name")

            def transfer():
                url = f"https://huggingface.co/datasets/{self.repo}/resolve/{rev}/{self.namespace}/{obj}"
                with self.http.get(url, timeout=(15, 120), stream=True) as response:
                    response.raise_for_status()

                    def write(handle):
                        h = hashlib.sha256()
                        length = 0
                        for chunk in response.iter_content(1024 * 1024):
                            handle.write(chunk)
                            h.update(chunk)
                            length += len(chunk)
                        if length != meta["bytes"] or h.hexdigest() != meta["sha256"]:
                            raise RuntimeError("Pinned download failed its digest")

                    write_atomic(target, write)

            self.call(transfer)

    def reconcile(self, unit, store, selected=None):
        remote, rev = self.selected(unit) if selected is None else selected
        if not remote:
            return
        if remote["identity"] != store.tag:
            raise RuntimeError("Remote identity differs from requested fit")
        local = store.pointer
        if local and local["generation"] >= remote["generation"]:
            name = f"ledger/{remote['generation']:06d}.json"
            if local["files"].get(name, {}).get("sha256") != remote["files"][name]["sha256"]:
                raise RuntimeError("Local and remote checkpoint histories diverge")
            return
        if local:
            name = f"ledger/{local['generation']:06d}.json"
            if remote["files"].get(name, {}).get("sha256") != local["files"][name]["sha256"]:
                raise RuntimeError("Remote checkpoint has a different history")
        names = [n for n in remote["files"] if n != "POINTER.json"] + ["POINTER.json"]
        self.fetch(unit, remote, rev, store.root, names)
        store.pointer = json_read(store.pointer_path)
        store.load()
        self.record("restored_verified_bytes", unit=unit, revision=rev, generation=remote["generation"])
