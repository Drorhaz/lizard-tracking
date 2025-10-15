#!/usr/bin/env python3
"""
Promote corrected samples from the review queue into the pose seed dataset.

Steps:
 1. Read `data/review_queue/queue.jsonl`.
 2. Copy the first N reviewed items (configurable) from `dataset/review_queue`
    into `dataset/pose-seed`.
 3. Optionally clear whatever remains in the queue so the semi-auto pipeline
    can repopulate it on the next pass.

Adjust the configuration block below to change promotion count or paths.
"""
from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Set


@dataclass
class PromoteConfig:
    manifest: Path = Path("data/review_queue/queue.jsonl") #review_queue
    queue_images: Path = Path("dataset/review_queue/images")
    queue_labels: Path = Path("dataset/review_queue/labels")
    pose_images: Path = Path("dataset/pose-seed/images")
    pose_labels: Path = Path("dataset/pose-seed/labels")
    promote_count: int = 50
    clear_remaining: bool = True
    dry_run: bool = False
    random_selection: bool = True
    random_seed: Optional[int] = None
    # Additional behaviour
    mode: str = "manifest_ref"  # options: 'manifest_ref' or 'random_from_pose'
    random_from_pose_dir: Path = Path("dataset") #pose
    random_promote_count: int = 100


CONFIG = PromoteConfig()


def read_manifest(manifest: Path) -> List[dict]:
    if not manifest.exists():
        print(f"[info] manifest not found: {manifest}")
        return []
    entries: List[dict] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                print(f"[warn] skipping malformed line: {line[:80]}")
                continue
            entries.append(payload)
    return entries


def relative_to_root(path: Path, root: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return Path(path.name)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def copy_pair(src_img: Path, src_lbl: Path, dst_img: Path, dst_lbl: Path) -> None:
    ensure_parent(dst_img)
    ensure_parent(dst_lbl)
    shutil.copy2(src_img, dst_img)
    shutil.copy2(src_lbl, dst_lbl)


def remove_artifacts(paths: Iterable[Path]) -> None:
    for path in paths:
        if path.exists():
            path.unlink()
        bak = path.with_suffix(path.suffix + ".bak")
        if bak.exists():
            bak.unlink()


def prune_empty_dirs(root: Path, limit: Path) -> None:
    # Remove empty directories beneath root up to the specified limit (exclusive).
    for path in sorted({p.parent for p in root.rglob("*")}, key=lambda p: len(p.parts), reverse=True):
        if limit in path.parents or path == limit:
            continue
        try:
            path.rmdir()
        except OSError:
            pass


def main() -> None:
    cfg = CONFIG
    promoted = 0
    remaining_entries: List[dict] = []

    if cfg.mode == "random_from_pose":
        # gather labeled images from the configured pose dir
        pose_root = cfg.random_from_pose_dir
        candidates: List[Path] = []
        for p in (pose_root / "images").rglob("*"):
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
                continue
            lbl = (pose_root / "labels") / p.relative_to(pose_root / "images").with_suffix(".txt")
            if not lbl.exists():
                continue
            # ensure label has content
            txt = lbl.read_text(encoding="utf-8", errors="ignore").strip()
            if not txt:
                continue
            candidates.append((p, lbl))

        total_available = len(candidates)
        to_promote = min(cfg.random_promote_count, total_available) if cfg.random_promote_count > 0 else total_available
        if to_promote <= 0:
            print("[info] no candidates found for random_from_pose mode.")
            return
        rng = random.Random(cfg.random_seed)
        chosen = rng.sample(candidates, to_promote)
        for img_path, lbl_path in chosen:
            rel = relative_to_root(img_path, cfg.queue_images)
            dst_img = cfg.pose_images / rel
            dst_lbl = cfg.pose_labels / rel.with_suffix(".txt")
            if cfg.dry_run:
                print(f"[dry] would copy {img_path} -> {dst_img}")
            else:
                copy_pair(img_path, lbl_path, dst_img, dst_lbl)
            promoted += 1

    else:
        # manifest_ref: use queue.jsonl as a list of names, but copy data from the pose folder
        entries = read_manifest(cfg.manifest)
        if not entries:
            print("[info] nothing to promote; manifest empty.")
            return
        
        total_available = len(entries)
        to_promote = min(cfg.promote_count, total_available) if cfg.promote_count > 0 else total_available
        selected_indices: Set[int] = set()
        if to_promote > 0:
            if cfg.random_selection:
                rng = random.Random(cfg.random_seed)
                selected_indices = set(rng.sample(range(total_available), to_promote))
            else:
                selected_indices = set(range(to_promote))

        for idx, entry in enumerate(entries):
            if idx not in selected_indices:
                remaining_entries.append(entry)
                continue

            img_path = Path(entry.get("image") or "")
            lbl_path = Path(entry.get("label") or "")
            # When manifest_ref mode is active we prefer to copy the authoritative
            # labeled files from the pose directory if they exist there.
            pose_src_img = cfg.pose_images / relative_to_root(img_path, cfg.queue_images)
            pose_src_lbl = cfg.pose_labels / relative_to_root(lbl_path, cfg.queue_labels).with_suffix(".txt")
            use_img = pose_src_img if pose_src_img.exists() else img_path
            use_lbl = pose_src_lbl if pose_src_lbl.exists() else lbl_path

            if not use_img.exists() or not use_lbl.exists():
                print(f"[skip] missing files for {entry.get('image')}; leaving in queue.")
                remaining_entries.append(entry)
                continue
            if cfg.dry_run:
                print(f"[dry] promote {use_img} -> pose-seed")
            else:
                rel = relative_to_root(use_img, cfg.queue_images)
                dst_img = cfg.pose_images / rel
                dst_lbl = cfg.pose_labels / rel.with_suffix(".txt")
                copy_pair(use_img, use_lbl, dst_img, dst_lbl)
                remove_artifacts([use_img, use_lbl])
            promoted += 1

    if cfg.clear_remaining and not cfg.dry_run:
        for entry in remaining_entries:
            img_path = Path(entry.get("image") or "")
            lbl_path = Path(entry.get("label") or "")
            remove_artifacts([img_path, lbl_path])
        cfg.manifest.unlink(missing_ok=True)
        prune_empty_dirs(cfg.queue_images, cfg.queue_images)
        prune_empty_dirs(cfg.queue_labels, cfg.queue_labels)
        print(f"[clean] removed remaining {len(remaining_entries)} queue samples.")
    else:
        if cfg.dry_run:
            print(f"[dry] would rewrite manifest with {len(remaining_entries)} entries.")
        else:
            with cfg.manifest.open("w", encoding="utf-8") as handle:
                for entry in remaining_entries:
                    handle.write(json.dumps(entry) + "\n")
        print(f"[info] {len(remaining_entries)} entries left in queue.")

    print(f"[done] promoted {promoted} samples to pose-seed.")


if __name__ == "__main__":
    main()
