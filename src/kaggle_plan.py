"""Fixed ownership of the 78 campaign fits, shared with the dashboard builder."""


def worker_plan(rows, num_workers):
    """Keep every shared bank and both consuming architectures on the same worker."""
    if num_workers != 4:
        raise ValueError("This campaign is pinned to four worker IDs.")
    owners = {}
    load = [0] * num_workers
    teacher_owner = {"cifar100": 0, "cifar10": 1, "tiny_imagenet200": 2}
    blocks = {}
    for row in rows:
        if row["role"] == "teacher" or row["arm"] == "hard_labels":
            w = teacher_owner[row["dataset"]] if row["role"] == "teacher" else 3
            owners[row["run_id"]] = w
            load[w] += int(row["updates"])
        else:
            blocks.setdefault((row["dataset"], row["seed"]), []).append(row)

    def cost(block):
        return sum(int(r["updates"]) for r in block)

    for key in sorted(blocks, key=lambda k: (-cost(blocks[k]), k)):
        w = min(range(num_workers), key=lambda i: (load[i], i))
        for r in blocks[key]:
            owners[r["run_id"]] = w
        load[w] += cost(blocks[key])
    return owners
