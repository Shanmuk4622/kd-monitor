"""Check the Kaggle image against the completed preflight before starting a worker."""


def check_worker_environment(hub, base):
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from kaggle_run import runtime_id
    from kd_fit_io import hash_object, json_read

    name = "qualification-cifar10"
    manifest, revision = hub.selected(name)
    if not manifest or not manifest["complete"]:
        raise RuntimeError(f"Missing completed {name} in {hub.repo}/{hub.namespace}. Check HF_REPO and NB01.")
    with TemporaryDirectory(dir=base) as tmp:
        hub.fetch(name, manifest, revision, tmp, ["qualification.json"])
        identity = json_read(Path(tmp) / "qualification.json")["identity"]
    if hash_object(identity) != manifest["identity"]:
        raise RuntimeError("Preflight receipt does not match its published identity.")
    expected = identity["environment"]
    actual = runtime_id()
    differences = [
        f"  {key}: preflight={expected.get(key)!r}, worker={actual.get(key)!r}"
        for key in sorted(expected.keys() | actual.keys()) if expected.get(key) != actual.get(key)
    ]
    if differences:
        raise RuntimeError(
            "Kaggle environment differs from the completed preflight:\n" + "\n".join(differences)
            + "\nCopy and edit the successful NB01 version, pin its original environment, "
            "import this worker notebook into the copy, choose GPU T4 x2, and Run All."
        )
    print("Worker environment matches preflight.", flush=True)
