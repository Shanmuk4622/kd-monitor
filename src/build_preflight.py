"""Regenerate the self-contained preflight notebook from the two readable sources."""

from pathlib import Path

import nbformat as nb


def main():
    root = Path(__file__).resolve().parents[1]
    notebook = nb.v4.new_notebook()
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.cells = [
        nb.v4.new_markdown_cell(
            "# KD view-allocation preflight\n\n"
            "Run All in a fresh kernel. A small synthetic CNN checks cache budgets, image/view "
            "alignment and distillation gradients on CPU and each available CUDA device. The "
            "optional final cell builds the real split manifest if the original CIFAR-100 binary "
            "archive is attached. The notebook writes local evidence and performs no benchmark fit."
        ),
        nb.v4.new_markdown_cell(
            "## Output directory and optional input\n\n"
            "On Kaggle, attach cifar-100-binary.tar.gz for the data audit. One matching "
            "archive is detected automatically. If several are attached, set CIFAR100_ARCHIVE "
            "below to the intended file. The synthetic checks require no data or secret."
        ),
        nb.v4.new_code_cell(
            'from pathlib import Path\n'
            'import os\n\n'
            'CIFAR100_ARCHIVE = None\n'
            'base = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()\n'
            'OUTPUT = Path(os.environ.get("KD_PREFLIGHT_OUTPUT", base / "kd_view_preflight"))\n'
            'print("Evidence directory:", OUTPUT)'
        ),
        nb.v4.new_markdown_cell("## Cache creation, allocation and replay checks"),
        nb.v4.new_code_cell((root / "src/cache_preflight.py").read_text(encoding="utf-8")),
        nb.v4.new_code_cell(
            'fixture_dir, receipt = run_fixture(OUTPUT)\n'
            'print("Fixture evidence:", fixture_dir)\n'
            'for device, rows in receipt["results"].items():\n'
            '    for row in rows:\n'
            '        print(device, row["policy"], "calls", row["standalone_queries"],\n'
            '              "max logit error", row["max_logit_error"],\n'
            '              "max gradient error", row["max_gradient_error"])'
        ),
        nb.v4.new_markdown_cell(
            "## Data roles and pixel-overlap audit\n\n"
            "The small split test always runs. The real audit verifies the official archive digest, "
            "exports training/validation labels, and withholds final-split labels. "
            "Keep the archive read-only and retain both the manifest and its receipt."
        ),
        nb.v4.new_code_cell((root / "src/cifar100_manifest.py").read_text(encoding="utf-8")),
        nb.v4.new_code_cell(
            'print("Synthetic split counts:", check_split_fixture())\n'
            'archive = Path(CIFAR100_ARCHIVE) if CIFAR100_ARCHIVE is not None else None\n'
            'if archive is None and Path("/kaggle/input").is_dir():\n'
            '    matches = sorted(Path("/kaggle/input").rglob("cifar-100-binary.tar.gz"))\n'
            '    if len(matches) > 1:\n'
            '        raise ValueError("Select one archive in CIFAR100_ARCHIVE.")\n'
            '    archive = matches[0] if matches else None\n'
            'if archive is None:\n'
            '    print("Real CIFAR-100 manifest pending: attach cifar-100-binary.tar.gz.")\n'
            'else:\n'
            '    split_dir, split_receipt = audit_cifar100(archive, OUTPUT)\n'
            '    print("CIFAR-100 manifest:", split_dir)\n'
            '    print(split_receipt)'
        ),
        nb.v4.new_markdown_cell(
            "The numerical checks concern the fixture CNN and its view cache. The measured run also "
            "requires a trained teacher, qualification of its own runtime and storage, and a successful "
            "Kaggle interruption and remote-restore test. See the project's RUN_GUIDE.md."
        ),
    ]
    for index, cell in enumerate(notebook.cells):
        cell.id = f"kd-preflight-{index:02d}"
    output = root / "notebooks/00_cache_preflight.ipynb"
    output.parent.mkdir(exist_ok=True)
    nb.validate(notebook)
    nb.write(notebook, output)


if __name__ == "__main__":
    main()
