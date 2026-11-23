"""Build six independently importable Kaggle notebooks with identical embedded source."""

import base64
import hashlib
import io
import zipfile
from pathlib import Path
import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "research"
OUT = ROOT / "notebooks"
# Only this reviewed bundle may import the preceding release's CIFAR capacity probes.
# A future runtime change falls back to qualifying the inputs itself.
CIFAR_COMPATIBILITY = {
    "44b34996bf45cc167a213a61467b66d268a2c0062c61c259cd94f11fc573c2f9":
        "4f73c5a747c3ca210a388ed5375a3b96be176d0245d2539fe45357d957d9c7e9",
}
STAGES = [
    (
        "01_Kaggle_Preflight",
        "Qualify the inputs, models and recovery",
        "qualify(hub,base,output,INPUT_PATHS,VERSIONS)",
        "Attach all three Kaggle datasets. Run this notebook first on two T4 GPUs. It publishes a disposable recovery anchor, dataset manifests and per-model capacity forecasts. Expect real work here: one disposable epoch for each of the nine dataset/model combinations. The last output confirms when to open notebook 02.",
    ),
    (
        "02_Kaggle_Train",
        "Train or resume the experiment",
        "train(hub,base,output,INPUT_PATHS,VERSIONS,GROUP,NUM_WORKERS,WORKER_ID)",
        "After preflight, launch each of the four training notebooks in its own Kaggle session. Every file has its worker ID set. Relaunch the same worker after interruption.",
    ),
    (
        "03_Kaggle_Results",
        "Score the frozen test sets and export paper results",
        "release(hub,base,output,INPUT_PATHS,VERSIONS)",
        "Run after all 75 student fits and their audits have completed. This checks the entire matrix before evaluating the final cohorts, publishes per-image logits, predictions and confusion matrices and writes accuracy.csv, paired.json, resources.csv and reuse.csv to the Kaggle output folder.",
    ),
]


def main():
    OUT.mkdir(exist_ok=True)
    sources = [
        *sorted((ROOT / "src").glob("kd_*.py")),
        *(
            ROOT / "src" / n
            for n in (
                "kaggle_inputs.py",
                "kaggle_run.py",
                "kaggle_results.py",
                "cache_preflight.py",
                "cifar100_manifest.py",
            )
        ),
    ]
    sources.extend(DATA / n for n in ("PROTOCOL.md", "KAGGLE_EXPERIMENT_PLAN.md", "RUN_MATRIX_V2.csv"))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sources:
            info = zipfile.ZipInfo(p.name, (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, p.read_text(encoding="utf-8").encode())
    body = buf.getvalue()
    digest = hashlib.sha256(body).hexdigest()
    encoded = base64.b64encode(body).decode()
    loader = f"""import base64, hashlib, io, os, sys, tempfile, zipfile
from pathlib import Path
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
payload = base64.b64decode("{encoded}")
assert hashlib.sha256(payload).hexdigest() == "{digest}"
runtime = Path(tempfile.gettempdir()) / "kd-runtime-{digest[:14]}"
runtime.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
    for name in bundle.namelist():
        dest = runtime / name
        value = bundle.read(name)
        if dest.exists() and dest.read_bytes() != value:
            raise RuntimeError("Incompatible extracted runtime. Restart in a clean session.")
        if not dest.exists(): dest.write_bytes(value)
sys.path.insert(0,str(runtime))
print("Verified experiment runtime", "{digest[:14]}")"""
    input_resolver = (ROOT / "src" / "kaggle_notebook_inputs.py").read_text(encoding="utf-8")
    settings = """HF_REPO = "Shanmuk4622/kd-crop-budget-4w-2026"
TINY_KAGGLE_SOURCE = "shanmuk4622/tinyimagenet200"
GROUP = "all"
NUM_WORKERS = 4
VERSIONS = {}  # File hashes pin each Kaggle input automatically
INPUT_PATHS = {}  # Usually detected. To override: {"cifar100": "/kaggle/input/.../dataset-cifar100-python"}
SCRATCH = "/kaggle/temp/kd_view_v2"  # Can also use /kaggle/working/kd_view_v2"""
    deps = """import importlib.util, subprocess, sys
packages = {"huggingface_hub":"huggingface_hub", "timm":"timm", "pynvml":"nvidia-ml-py", "scipy":"scipy", "filelock":"filelock"}
missing = [package for module,package in packages.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable,"-m","pip","install","--disable-pip-version-check",*missing])"""
    jobs = [(1, STAGES[0], None), *((2, STAGES[1], w) for w in range(4)), (3, STAGES[2], None)]
    for i, (filename, title, call, desc), worker in jobs:
        if i == 1:
            call = call[:-1] + f",previous_cifar_source={CIFAR_COMPATIBILITY.get(digest)!r})"
        role = "preflight" if i == 1 else "results"
        config = settings
        if worker is not None:
            role = f"worker-{worker}"
            if worker:
                filename += f"_Worker{worker}"
            title += f" · Worker {worker}"
            config += f"\nWORKER_ID = {worker}\nMONITOR_ENABLED = True"
        resolve = input_resolver + '''

from kaggle_inputs import SOURCES

SOURCES["tiny_imagenet200"] = TINY_KAGGLE_SOURCE
RESOLVED_PATHS = dict(INPUT_PATHS)
RESOLVED_PATHS["tiny_imagenet200"] = tiny_input_path(TINY_KAGGLE_SOURCE, INPUT_PATHS.get("tiny_imagenet200"))
print("Tiny ImageNet input:", RESOLVED_PATHS["tiny_imagenet200"])
'''
        run = (
            """from kaggle_run import check_settings, session, qualify, train
from kaggle_results import release
check_settings(HF_REPO, RESOLVED_PATHS, VERSIONS)
with session(HF_REPO, SCRATCH, ROLE) as (hub, base, output):
    result = """
            + call.replace("INPUT_PATHS", "RESOLVED_PATHS")
            + """
print(result)"""
        )
        monitor_cells = []
        if worker is not None:
            monitor_cells = [
                nb.v4.new_markdown_cell("### Live worker dashboard\n\nSee monitoring/SETUP.md for Supabase and Kaggle secrets. Monitoring starts after the worker acquires its training lease."),
                nb.v4.new_code_cell((ROOT / "monitoring" / "worker.py").read_text(encoding="utf-8")),
            ]
            run = run.replace("    result = train(", "    with monitor_worker(WORKER_ID, base, hub, MONITOR_ENABLED):\n        result = train(")
        run = (
            run.replace("SCRATCH, ROLE", "SCRATCH, " + repr(role))
            if worker is None
            else run.replace("SCRATCH, ROLE", 'SCRATCH, f"worker-{WORKER_ID}"')
        )
        doc = nb.v4.new_notebook(
            cells=[
                nb.v4.new_markdown_cell(
                    "# "
                    + title
                    + "\n\n"
                    + desc
                    + "\n\n**Setup:** GPU T4 ×2, Internet enabled, all three project inputs attached, and the Kaggle secret `HF_TOKEN` enabled for this notebook. Keep the token out of code. The repository and worker settings are filled in. Use **Run All**. The mounted files and split are pinned by their hashes. See KAGGLE_RUN_INSTRUCTIONS.md for links and recovery."
                ),
                nb.v4.new_code_cell(config),
                nb.v4.new_markdown_cell(
                    "### Runtime dependencies\n\nInstalls missing Python packages only. Image data are read exclusively from Kaggle mounts."
                ),
                nb.v4.new_code_cell(deps),
                nb.v4.new_markdown_cell(
                    "### Embedded experiment code\n\nThe self-contained bundle verifies its digest before import."
                ),
                nb.v4.new_code_cell(loader),
                nb.v4.new_markdown_cell("### Locate Tiny ImageNet\n\nSelect the dataset folder inside the Kaggle mount before qualification, training or scoring."),
                nb.v4.new_code_cell(resolve),
                *monitor_cells,
                nb.v4.new_markdown_cell("### Execute this stage"),
                nb.v4.new_code_cell(run),
            ]
        )
        doc.metadata.kernelspec = dict(display_name="Python 3", language="python", name="python3")
        for j, c in enumerate(doc.cells):
            c.id = f"kd-v2-{i}-{j}"
        nb.validate(doc)
        nb.write(doc, OUT / (filename + ".ipynb"))
    return digest


if __name__ == "__main__":
    print(main())
