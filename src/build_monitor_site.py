"""Build the Pages entry point and its inventory from the campaign's actual allocator."""

import csv
import json
import re
from pathlib import Path

from kaggle_plan import worker_plan


def main():
    root = Path(__file__).resolve().parents[1]
    site = root / "monitoring/site"
    with (root / "research/RUN_MATRIX_V2.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    owners = worker_plan(rows, 4)
    inventory = [
        dict(id=r["run_id"], worker=owners[r["run_id"]], dataset=r["dataset"], model=r["model"],
             arm="teacher" if r["role"] == "teacher" else r["arm"], seed=int(r["seed"]),
             epochs=int(r["epochs"]), updates=int(r["updates"])) for r in rows
    ]
    plan = "window.KD_PLAN = [\n" + ",\n".join("  "+json.dumps(r) for r in inventory) + "\n];\n"
    (site / "plan.js").write_text(plan, encoding="utf-8", newline="\n")
    html = (site / "index.html").read_text(encoding="utf-8")
    html = html.replace('<link rel="stylesheet" href="style.css">',
                        "<style>\n"+(site / "style.css").read_text(encoding="utf-8")+"\n</style>")
    scripts = []
    for url in re.findall(r'<script defer src="([^"]+)"></script>', html):
        html = html.replace(f'<script defer src="{url}"></script>', "")
        if url.startswith("https://"):
            scripts.append(f'<script src="{url}"></script>')
        else:
            scripts.append("<script>\n"+(site / url).read_text(encoding="utf-8")+"\n</script>")
    target = root / "docs/index.html"
    target.parent.mkdir(exist_ok=True)
    html = "\n".join(line.rstrip() for line in html.splitlines())+"\n"
    target.write_text(html.replace("</body>", "\n".join(scripts)+"\n</body>"), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
