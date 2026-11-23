"""Build the portable CIFAR-100 Run All master from the readable runtime and protocol."""

import hashlib
import json
from pathlib import Path

import nbformat as nb


def main():
    root = Path(__file__).resolve().parents[1]
    names = [
        "cache_preflight.py",
        "cifar100_manifest.py",
        "kd_fit_io.py",
        "kd_fit_data.py",
        "kd_fit_engine.py",
        "kd_meter.py",
        "kd_hub.py",
        "kd_review.py",
        "kd_qualify.py",
        "kd_campaign.py",
    ]
    sources = {name: (root / "src" / name).read_text(encoding="utf-8") for name in names}
    sources["PROTOCOL.md"] = (root / "research" / "PROTOCOL.md").read_text(encoding="utf-8")
    # Canonical LF bytes give Windows and Linux the same scientific source identity.
    hashes = {n: hashlib.sha256(body.encode()).hexdigest() for n, body in sources.items()}
    payload = json.dumps(sources, ensure_ascii=True, sort_keys=True)
    extractor = (
        "from pathlib import Path\nimport hashlib, json, os, subprocess, sys\n"
        "from tempfile import gettempdir\n"
        "os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')\n"
        f"SOURCES = {payload}\nEXPECTED = {hashes!r}\n"
        "build = hashlib.sha256(json.dumps(EXPECTED,sort_keys=True).encode()).hexdigest()[:16]\n"
        "runtime = Path(gettempdir()) / ('kd-view-runtime-' + build)\nruntime.mkdir(parents=True,exist_ok=True)\n"
        "for name, source in SOURCES.items():\n"
        "    body=source.encode('utf-8')\n"
        "    if hashlib.sha256(body).hexdigest()!=EXPECTED[name]: raise RuntimeError('Embedded source mismatch')\n"
        "    target=runtime/name\n"
        "    if target.exists() and target.read_bytes()!=body: raise RuntimeError('Runtime identity conflict')\n"
        "    if not target.exists(): target.write_bytes(body)\n"
        "sys.path.insert(0,str(runtime))\n"
        "print('Verified embedded files',len(SOURCES),'build',build)"
    )
    doc = nb.v4.new_notebook()
    doc.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    doc.cells = [
        nb.v4.new_markdown_cell(
            "# CIFAR-100 knowledge distillation study\n\nRun All in a fresh kernel. "
            "Qualification exercises six student configurations, a deliberate crash and a fresh-process restart on disposable data. "
            "Measured mode runs the scientific teacher and the chosen seed blocks. Final review opens the test labels only after all 30 primary student runs and their evidence audits exist."
        ),
        nb.v4.new_code_cell(
            'MODE = "qualification"  # qualification, measured, final_review\n'
            'HF_DATASET_REPO = ""  # Existing dedicated owner/repo, enabled for the Kaggle HF_TOKEN secret\n'
            "CIFAR100_ARCHIVE = None  # Auto-detect one original cifar-100-binary.tar.gz in /kaggle/input\n"
            "SEED_BLOCKS = (42,)  # All primary blocks: (42, 123, 2026, 31415, 27182)"
        ),
        nb.v4.new_markdown_cell(
            "## Embedded, verified runtime\n\nThe notebook carries all experiment sources and the protocol. No source repository or model download is required."
        ),
        nb.v4.new_code_cell(extractor),
        nb.v4.new_markdown_cell(
            "## Run the selected stage\n\nFor measured mode, attach the original CIFAR-100 binary archive, select two T4 GPUs and enable Internet and the Kaggle HF_TOKEN secret. "
            "The runner verifies remote continuation and local capacity before fitting. Preserve the executed notebook and its output receipts."
        ),
        nb.v4.new_code_cell(
            "from kd_campaign import run_master\n"
            "base = Path(os.environ.get('KD_MASTER_OUTPUT', '/kaggle/working/kd_view_primary' if Path('/kaggle/working').is_dir() else 'kd_view_primary'))\n"
            "archive = Path(CIFAR100_ARCHIVE) if CIFAR100_ARCHIVE else None\n"
            "if MODE != 'qualification' and archive is None:\n"
            "    entries = sorted(Path('/kaggle/input').rglob('cifar-100-binary.tar.gz'))\n"
            "    if len(entries) != 1: raise ValueError('Set CIFAR100_ARCHIVE to the original binary archive')\n"
            "    archive = entries[0]\n"
            "result = run_master(MODE,base,archive,HF_DATASET_REPO,SEED_BLOCKS,EXPECTED['PROTOCOL.md'])\n"
            "print(json.dumps(result,indent=2))"
        ),
    ]
    for i, cell in enumerate(doc.cells):
        cell.id = f"kd-master-{i:02d}"
    nb.validate(doc)
    nb.write(doc, root / "notebooks/01_cifar100_master.ipynb")


if __name__ == "__main__":
    main()
