"""Fixed-schedule training shared by the CIFAR teacher and its six student arms."""

import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from kd_fit_data import fresh_codes, source_order
from kd_fit_io import csv_write, json_write, np_write, rng_restore, rng_state, seed_all, torch_write


def model_for(name, classes):
    if name == "fixture":
        return nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(8, classes),
        )
    if os.name == "nt":
        dlls = Path(sys.prefix) / "Library/bin"
        if dlls.exists():
            global _dll_handle
            _dll_handle = os.add_dll_directory(str(dlls))
        from PIL import Image  # noqa: F401 - load the Windows imaging DLL before torchvision.
    from torchvision import models

    if name == "mobilenetv4_conv_small":
        import timm

        net = timm.create_model(name, pretrained=False, num_classes=classes, drop_rate=0, drop_path_rate=0)
        net.conv_stem.stride = (1, 1)
        return net
    if name not in ("resnet18", "resnet34"):
        raise ValueError("Unsupported CIFAR-100 primary model")
    net = getattr(models, name)(weights=None, num_classes=classes)
    net.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    net.maxpool = nn.Identity()
    return net


def lr_for(update, total, warmup):
    if update < warmup:
        return 0.1 * (update + 1) / warmup
    return 0.05 * (1.0 + math.cos(math.pi * (update - warmup) / max(total - warmup - 1, 1)))


def evaluate(model, data, ids, device, batch):
    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(ids), batch):
            idx = ids[start : start + batch]
            features, _ = data.inputs(idx, None, device)
            chunks.append(model(features).cpu().numpy())
    logits = np.concatenate(chunks)
    labels = data.labels[ids]
    if not np.isfinite(logits).all() or not len(labels):
        raise ValueError("Invalid evaluation predictions or empty scoring cohort")
    nll = F.cross_entropy(torch.from_numpy(logits).double(), torch.from_numpy(labels)).item()
    return logits, int(np.sum(logits.argmax(1) == labels)), nll


def train_fit(
    store,
    data,
    config,
    device,
    on_commit,
    stop_requested=lambda: False,
    teacher=None,
    bank=None,
    until=None,
    fault_update=None,
):
    seed, arm, epochs, batch = (config[k] for k in ("seed", "arm", "epochs", "batch"))
    seed_all(seed)
    model = model_for(config["model"], data.classes).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=False, foreach=False
    )
    state = store.load()
    steps = math.ceil(len(data.fit) / batch)
    total = epochs * steps
    warm = config["warm_epochs"] * steps
    if state:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        rng_restore(state["rng"])
        step, best, partial = state["update"], state["best"], state["partial"]
        if state["complete"]:
            return state
    else:
        step, best, partial = 0, None, [0, 0.0, 0.0, 0.0, 0]
        json_write(
            store.path("config.json"),
            {
                **config,
                "contract": store.identity,
                "mean": data.mean.tolist(),
                "std": data.std.tolist(),
                "optimizer": "SGD",
                "scaler": None,
                "learning_rate": "linear warm-up then cosine by optimizer update",
            },
        )
        initial = dict(
            update=0,
            complete=False,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            rng=rng_state(),
            best=None,
            partial=partial,
            scheduler_update=0,
            scaler=None,
        )
        store.commit(initial, ["config.json"])
        on_commit(store, True)
    if arm == "live_kd" and teacher is None:
        raise ValueError("Live KD needs the selected teacher")
    if arm not in ("teacher", "hard_labels", "live_kd") and bank is None:
        raise ValueError("Cached KD needs a completed matching bank")
    if teacher is not None:
        teacher.eval().requires_grad_(False)
    exposure, update_rows, logits_parts, live_parts, new_files = [], [], [], [], []
    sample_offset = 0
    while step < total:
        epoch, bindex = divmod(step, steps)
        if not update_rows or bindex == 0:
            order = source_order(len(data.fit), seed, epoch)
            fresh = fresh_codes(data, seed, epoch)
        positions = order[bindex * batch : (bindex + 1) * batch]
        indices = data.fit[positions]
        slots = np.full(len(indices), -1, dtype=np.int64)
        codes = fresh[positions]
        target = None
        if bank is not None:
            slots = (epoch + bank["slot_offsets"][positions]) % bank["counts"][positions]
            codes = bank["codes"][positions, slots]
            target = torch.from_numpy(np.array(bank["logits"][bank["offsets"][positions] + slots])).to(
                device
            )
        features, y = data.inputs(indices, codes, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        rate = lr_for(step, total, warm)
        optimizer.param_groups[0]["lr"] = rate
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        begin = time.perf_counter()
        if arm == "live_kd":
            with torch.no_grad():
                target = teacher(features)
            live_parts.append(target.cpu().numpy())
        preds = model(features)
        ce = F.cross_entropy(preds, y, reduction="none")
        kl = None
        if target is None:
            obj = ce.mean()
        else:
            kl = F.kl_div((preds / 4).log_softmax(-1), (target / 4).softmax(-1), reduction="none").sum(1)
            obj = (0.5 * ce + 8 * kl).mean()
        if not torch.isfinite(obj):
            raise FloatingPointError(f"Nonfinite objective at update {step}")
        obj.backward()
        norm = torch.linalg.vector_norm(
            torch.stack([p.grad.detach().norm() for p in model.parameters() if p.grad is not None])
        )
        if not torch.isfinite(norm):
            raise FloatingPointError(f"Nonfinite gradient at update {step}")
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - begin
        logits = preds.detach().cpu().numpy()
        probabilities = preds.detach().softmax(1).gather(1, y[:, None]).ravel().cpu().numpy()
        prediction = logits.argmax(1)
        labels = data.labels[indices]
        count = len(indices)
        correct = int(np.sum(prediction == labels))
        ces, kls, objs = (
            ce.detach().sum().item(),
            (kl.detach().sum().item() if kl is not None else 0.0),
            obj.item() * count,
        )
        partial = [
            partial[0] + count,
            partial[1] + ces,
            partial[2] + kls,
            partial[3] + objs,
            partial[4] + correct,
        ]
        update_rows.append(
            (
                step + 1,
                epoch + 1,
                bindex,
                count,
                ces,
                kls if kl is not None else "",
                objs,
                correct,
                rate,
                norm.item(),
                seconds,
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else "",
            )
        )
        exposure.extend(
            (
                step + 1,
                step,
                epoch + 1,
                sample_offset + j,
                data.ids[i],
                int(labels[j]),
                int(codes[j]),
                int(slots[j]) if bank is not None else "",
                int(prediction[j]),
                float(probabilities[j]),
            )
            for j, i in enumerate(indices)
        )
        logits_parts.append(logits)
        sample_offset += count
        step += 1
        boundary = step % steps == 0
        if boundary:
            val, val_correct, nll = evaluate(model, data, data.validation, device, batch)
            folder = f"epochs/{epoch+1:03d}"
            weights = folder + "/weights.pt"
            torch_write(store.path(weights), model.state_dict())
            np_write(store.path(folder + "/validation_logits.npy"), val)
            csv_write(
                store.path(folder + "/validation.csv"),
                ["row", "image_id", "label", "prediction"],
                ((j, data.ids[i], data.labels[i], val[j].argmax()) for j, i in enumerate(data.validation)),
            )
            csv_write(
                store.path(folder + "/metrics.csv"),
                [
                    "epoch",
                    "fit_n",
                    "ce_sum",
                    "kl_sum",
                    "objective_sum",
                    "fit_correct",
                    "validation_n",
                    "validation_correct",
                    "validation_nll",
                ],
                [
                    (
                        epoch + 1,
                        *partial[:2],
                        partial[2] if target is not None else "",
                        *partial[3:],
                        len(data.validation),
                        val_correct,
                        nll,
                    )
                ],
            )
            new_files.extend(
                [
                    weights,
                    folder + "/validation_logits.npy",
                    folder + "/validation.csv",
                    folder + "/metrics.csv",
                ]
            )
            if best is None or (val_correct, -nll) > (best["correct"], -best["nll"]):
                best = dict(epoch=epoch + 1, correct=val_correct, nll=nll, weights=weights)
            partial = [0, 0.0, 0.0, 0.0, 0]
        should_stop = (until is not None and step >= until) or stop_requested()
        if boundary or len(update_rows) >= config["commit_updates"] or should_stop:
            folder = f"updates/{step:08d}"
            csv_write(
                store.path(folder + "/updates.csv"),
                [
                    "update",
                    "epoch",
                    "batch",
                    "n",
                    "ce_sum",
                    "kl_sum",
                    "objective_sum",
                    "correct",
                    "learning_rate",
                    "gradient_l2",
                    "seconds",
                    "peak_allocated_bytes",
                ],
                update_rows,
            )
            csv_write(
                store.path(folder + "/exposure.csv"),
                [
                    "update",
                    "weight_version",
                    "epoch",
                    "logit_row",
                    "image_id",
                    "label",
                    "crop_flip_code",
                    "bank_slot",
                    "prediction",
                    "true_probability",
                ],
                exposure,
            )
            np_write(store.path(folder + "/student_logits.npy"), np.concatenate(logits_parts))
            new_files.extend(
                folder + "/" + n for n in ("updates.csv", "exposure.csv", "student_logits.npy")
            )
            if live_parts:
                name = folder + "/live_teacher_logits.npy"
                np_write(store.path(name), np.concatenate(live_parts))
                new_files.append(name)
            state = dict(
                update=step,
                complete=step == total,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                rng=rng_state(),
                best=best,
                partial=partial,
                scheduler_update=step,
                scaler=None,
            )
            if state["complete"]:
                json_write(
                    store.path("ENDPOINT.json"),
                    dict(
                        final=f"epochs/{epochs:03d}/weights.pt",
                        best=best,
                        update=step,
                        fit_images=len(data.fit),
                        validation_images=len(data.validation),
                    ),
                )
                new_files.append("ENDPOINT.json")
            store.commit(state, new_files, fault="before_pointer" if step == fault_update else None)
            on_commit(store, state["complete"] or should_stop)
            exposure, update_rows, logits_parts, live_parts, new_files = [], [], [], [], []
            sample_offset = 0
            if boundary:
                print(
                    f"{config['model']} {arm} epoch {epoch+1}/{epochs}, durable update {step}", flush=True
                )
            if should_stop:
                return state
    return state
