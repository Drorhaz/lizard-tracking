#!/usr/bin/env python3
"""
Semi-automatic pose labeling pipeline.

Workflow:
1. Train a fast YOLO pose model on the current seed dataset.
2. Infer remaining frames, routing confident ones back into the seed set and
   queueing the rest for manual review.
3. Overwrite the rolling `best.pt` so subsequent runs always resume from the
   freshest checkpoint.

Adjust the configuration block below to tune directories, thresholds, and training knobs.
"""
from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
import random
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from ultralytics import YOLO

try:
    import torch
except ImportError:  # pragma: no cover - torch shipped with ultralytics, but guard anyway
    torch = None


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass
class PipelineConfig:
    # Core paths
    seed_dataset: Path = Path("dataset/pose-seed")
    unlabeled_dataset: Path = Path("dataset/head-detection-dataset")
    review_queue: Path = Path("dataset/review_queue")
    rolling_best: Path = Path("autogenerate/best.pt")
    base_weights: Path = Path("yolo11n-pose.pt")
    dataset_yaml: Path = Path("autogenerate/pose_seed.yaml")
    training_project_dir: Path = Path("autogenerate/runs_pose_seed")
    queue_manifest: Path = Path("data/review_queue/queue.jsonl")

    # Training hyper-params
    epochs: int = 20
    learning_rate: float = 0.035  # intentionally high for rapid adaptation
    batch: int = 16
    imgsz: int = 640
    device: str = "auto"
    patience: int = 5
    augment: bool = True

    # Routing thresholds
    predict_conf: float = 0.2
    high_conf_box: float = 0.8
    high_conf_kpt: float = 0.7
    # Lower thresholds for review queue (middle range gets skipped)
    low_conf_box: float = 0.4
    low_conf_kpt: float = 0.3

    # Dataset metadata
    class_name: str = "lizard_head"
    num_keypoints: int = 3
    keypoint_dims: int = 3  # (x, y, conf)
    splits: Tuple[str, ...] = ("train", "val")

    # Behaviour toggles
    overwrite_existing: bool = False
    dry_run: bool = False
    clear_caches: bool = True
    # Seed trimming (keep only a fixed number of labeled samples) - DISABLED
    trim_seed: bool = False  # Disabled - keep all samples regardless of count
    max_seed_images: int = 100
    seed_train_count: int = 70
    seed_val_count: int = 30
    random_seed: Optional[int] = None

    def resolve_yaml_contents(self) -> str:
        """Create the on-disk YAML expected by Ultralytics."""
        seed_abs = self.seed_dataset.resolve()
        return (
            f"path: {seed_abs}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"\n"
            f"nc: 1\n"
            f"names:\n"
            f"  0: {self.class_name}\n"
            f"kpt_shape: [{self.num_keypoints}, {self.keypoint_dims}]\n"
        )


CONFIG = PipelineConfig(
    # Core paths
    seed_dataset=Path("dataset/pose-seed"),
    unlabeled_dataset=Path("dataset/head-detection-dataset"),
    review_queue=Path("dataset/review_queue"),
    rolling_best=Path("autogenerate/best.pt"),
    base_weights=Path("yolo11n-pose.pt"),
    dataset_yaml=Path("autogenerate/pose_seed.yaml"),
    training_project_dir=Path("autogenerate/runs_pose_seed"),
    queue_manifest=Path("data/review_queue/queue.jsonl"),

    # Training hyper-params
    epochs=20,
    learning_rate=0.035,
    batch=16,
    imgsz=640,
    device="auto",
    patience=5,
    augment=True,

    # Routing thresholds
    predict_conf=0.2,
    high_conf_box=0.8,
    high_conf_kpt=0.7,
    low_conf_box=0.4,
    low_conf_kpt=0.3,

    # Dataset metadata
    class_name="lizard_head",
    num_keypoints=3,
    keypoint_dims=3,
    splits=("train", "val"),

    # Behaviour toggles
    overwrite_existing=False,
    dry_run=False,  # Set to False to actually train and infer
    
    # Seed trimming (keep only a fixed number of labeled samples) - DISABLED
    trim_seed=False,  # Disabled - keep all samples regardless of count
    max_seed_images=100,
    seed_train_count=70,
    seed_val_count=30,
    random_seed=None,
)


def ensure_structure(cfg: PipelineConfig) -> None:
    """Ensure required folders and the training YAML exist."""
    for root in (cfg.seed_dataset, cfg.review_queue):
        for split in cfg.splits:
            (root / "images" / split).mkdir(parents=True, exist_ok=True)
            (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    cfg.training_project_dir.mkdir(parents=True, exist_ok=True)
    cfg.dataset_yaml.parent.mkdir(parents=True, exist_ok=True)
    if not cfg.dataset_yaml.exists() or cfg.overwrite_existing:
        cfg.dataset_yaml.write_text(cfg.resolve_yaml_contents(), encoding="utf-8")

    cfg.queue_manifest.parent.mkdir(parents=True, exist_ok=True)

    if cfg.clear_caches:
        for cache_path in (cfg.seed_dataset / "images").rglob("*.cache"):
            try:
                cache_path.unlink()
                print(f"[prep] removed stale cache {cache_path}")
            except FileNotFoundError:
                continue


def prune_empty_dirs(root: Path, limit: Path) -> None:
    # Remove empty directories beneath root up to the specified limit (exclusive).
    for path in sorted({p.parent for p in root.rglob("*")}, key=lambda p: len(p.parts), reverse=True):
        if limit in path.parents or path == limit:
            continue
        try:
            path.rmdir()
        except OSError:
            pass


def trim_pose_seed(cfg: PipelineConfig) -> None:
    """Keep only a limited number of labeled samples in the seed dataset.

    Keeps up to `cfg.max_seed_images` images total, with `seed_train_count` and
    `seed_val_count` used as the desired split. Only files with non-empty
    labels are considered. Excess files are removed (images, labels, caches).
    """
    if not cfg.trim_seed:
        return

    # collect labeled samples
    samples = []  # list of (img_path, lbl_path, rel, split)
    for split in cfg.splits:
        img_root = cfg.seed_dataset / "images" / split
        lbl_root = cfg.seed_dataset / "labels" / split
        if not img_root.exists():
            continue
        for img in img_root.rglob("*"):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            rel = img.relative_to(img_root)
            lbl = lbl_root / rel.with_suffix(".txt")
            if not lbl.exists():
                continue
            text = lbl.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue
            samples.append((img, lbl, rel, split))

    total = len(samples)
    keep_total = min(cfg.max_seed_images, total)
    if total <= keep_total:
        print(f"[trim] seed contains {total} labeled samples; nothing to trim.")
        return

    rng = random.Random(cfg.random_seed)
    # decide samples to keep while trying to respect desired split counts
    rng.shuffle(samples)
    # Prefer to keep up to seed_train_count from train and seed_val_count from val
    train_keep = cfg.seed_train_count
    val_keep = cfg.seed_val_count
    kept = []
    train_samples = [s for s in samples if s[3] == "train"]
    val_samples = [s for s in samples if s[3] == "val"]

    kept.extend(train_samples[:train_keep])
    kept.extend(val_samples[:val_keep])

    # If we don't yet have enough (or we overshot), adjust
    if len(kept) < keep_total:
        remaining = [s for s in samples if s not in kept]
        kept.extend(remaining[: keep_total - len(kept)])
    elif len(kept) > keep_total:
        # trim the excess from the end
        kept = kept[:keep_total]

    kept_set = { (p[0].resolve(), p[2], p[3]) for p in kept }

    removed = 0
    for img, lbl, rel, split in samples:
        key = (img.resolve(), rel, split)
        if key in kept_set:
            continue
        try:
            img.unlink()
        except FileNotFoundError:
            pass
        try:
            lbl.unlink()
        except FileNotFoundError:
            pass
        removed += 1

    # remove any .cache files under images
    for cache_path in (cfg.seed_dataset / "images").rglob("*.cache"):
        try:
            cache_path.unlink()
            print(f"[trim] removed cache {cache_path}")
        except FileNotFoundError:
            continue

    # prune empty directories beneath images/labels
    prune_empty_dirs(cfg.seed_dataset / "images", cfg.seed_dataset / "images")
    prune_empty_dirs(cfg.seed_dataset / "labels", cfg.seed_dataset / "labels")

    print(f"[trim] removed {removed} excess seed samples; kept {len(kept)}.")


def label_has_content(label_path: Path) -> bool:
    if not label_path.exists():
        return False
    text = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return False
    tokens = text.replace("\n", " ").split()
    if len(tokens) <= 1:
        return False
    try:
        values = [float(tok) for tok in tokens[1:]]
    except ValueError:
        return True
    return any(abs(v) > 1e-6 for v in values)

def pick_starting_weights(cfg: PipelineConfig) -> Path:
    """Return the checkpoint to fine-tune from."""
    if cfg.rolling_best.exists():
        return cfg.rolling_best
    if cfg.base_weights.exists():
        return cfg.base_weights
    return cfg.base_weights  # fall back to allowing Ultralytics to download


def resolve_device(cfg: PipelineConfig) -> str:
    """Convert `auto` into a concrete device string for Ultralytics."""
    if cfg.device != "auto":
        return cfg.device
    if torch is not None and torch.cuda.is_available():
        return "0"  # prefer first GPU when available
    return "cpu"


def train_pose_model(cfg: PipelineConfig) -> Path:
    """Train (or fine-tune) the YOLO pose model on the seed dataset."""
    weights_path = pick_starting_weights(cfg)
    print(f"[train] starting from weights: {weights_path}")

    if not weights_path.exists() and weights_path == cfg.base_weights:
        print(
            f"[train] base weights {weights_path} not found locally, "
            "Ultralytics will attempt to download."
        )

    try:
        model = YOLO(str(weights_path))
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[error] failed to load weights from {weights_path}: {exc}")
        sys.exit(1)

    run_name = f"pose_seed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if cfg.dry_run:
        print("[train] dry-run enabled; skipping actual training.")
        return cfg.rolling_best if cfg.rolling_best.exists() else Path()

    device = resolve_device(cfg)
    print(f"[train] resolved device: {device}")

    train_results = model.train(
        data=str(cfg.dataset_yaml),
        epochs=cfg.epochs,
        lr0=cfg.learning_rate,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device=device,
        project=str(cfg.training_project_dir),
        name=run_name,
        exist_ok=True,
        patience=cfg.patience,
        augment=cfg.augment,
        verbose=True,
    )

    save_dir = Path(getattr(model, "trainer", None).save_dir if hasattr(model, "trainer") else "")
    if not save_dir:
        save_dir = Path(train_results.save_dir) if hasattr(train_results, "save_dir") else None
    if not save_dir:
        save_dir = cfg.training_project_dir / run_name

    best_ckpt = save_dir / "weights" / "best.pt"
    if not best_ckpt.exists():
        print(f"[error] training finished but {best_ckpt} is missing.")
        sys.exit(1)

    cfg.rolling_best.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_ckpt, cfg.rolling_best)
    print(f"[train] updated rolling checkpoint -> {cfg.rolling_best}")
    return cfg.rolling_best


def iter_unlabeled_images(cfg: PipelineConfig) -> Iterable[Tuple[Path, Path, str]]:
    """Yield (absolute_image_path, relative_path_within_split, split)."""
    for split in cfg.splits:
        root = cfg.unlabeled_dataset / "images" / split
        if not root.exists():
            continue
        for image_path in root.rglob("*"):
            if image_path.suffix.lower() not in IMG_EXTS:
                continue
            rel = image_path.relative_to(root)
            yield image_path, rel, split


def detection_to_label_line(result, det_idx: int, cfg: PipelineConfig) -> Optional[str]:
    boxes = result.boxes
    keypoints = getattr(result, "keypoints", None)
    if boxes is None or len(boxes) == 0 or keypoints is None:
        return None

    xywhn = boxes.xywhn[det_idx].cpu().tolist()
    # keypoints.xyn is normalized [0, 1]; keypoints.data has (x, y, conf)
    kpts_xy = keypoints.xyn[det_idx].cpu().tolist()
    kpts_scores = keypoints.data[det_idx].cpu().tolist()

    line_parts = [
        "0",
        f"{xywhn[0]:.6f}",
        f"{xywhn[1]:.6f}",
        f"{xywhn[2]:.6f}",
        f"{xywhn[3]:.6f}",
    ]
    for i in range(cfg.num_keypoints):
        x = kpts_xy[i][0]
        y = kpts_xy[i][1]
        conf = kpts_scores[i][2]
        line_parts.append(f"{x:.6f}")
        line_parts.append(f"{y:.6f}")
        line_parts.append(f"{conf:.6f}")
    return " ".join(line_parts)


def classify_confidence(result, det_idx: int, cfg: PipelineConfig) -> Tuple[str, float, List[float]]:
    """Classify detection confidence as 'high', 'low', or 'skip'.
    
    Returns:
        - 'high': route to pose-seed 
        - 'low': route to review queue
        - 'skip': ignore (middle confidence range)
    """
    boxes = result.boxes
    keypoints = getattr(result, "keypoints", None)
    if boxes is None or len(boxes) == 0 or keypoints is None:
        return 'skip', 0.0, []

    box_conf = float(boxes.conf[det_idx].item())
    kpt_scores = keypoints.data[det_idx][:, 2].cpu().tolist()
    
    # High confidence: both box and all keypoints meet high thresholds
    if box_conf >= cfg.high_conf_box and all(score >= cfg.high_conf_kpt for score in kpt_scores):
        return 'high', box_conf, kpt_scores
    
    # Low confidence: either box or any keypoint is below low threshold
    elif box_conf < cfg.low_conf_box or any(score < cfg.low_conf_kpt for score in kpt_scores):
        return 'low', box_conf, kpt_scores
    
    # Middle range: skip
    else:
        return 'skip', box_conf, kpt_scores


def is_high_confidence(result, det_idx: int, cfg: PipelineConfig) -> Tuple[bool, float, List[float]]:
    """Legacy function - now uses classify_confidence internally."""
    conf_class, box_conf, kpt_scores = classify_confidence(result, det_idx, cfg)
    return conf_class == 'high', box_conf, kpt_scores


def route_predictions(cfg: PipelineConfig, weights_path: Path) -> None:
    if not weights_path.exists():
        print(f"[infer] rolling checkpoint {weights_path} not found; skipping inference.")
        return

    if cfg.dry_run:
        print("[infer] dry-run enabled; skipping inference.")
        return

    model = YOLO(str(weights_path))
    total = 0
    routed_to_pose_seed = 0  # High confidence + negative samples
    routed_to_review = 0     # Low confidence + no detection on regular images
    skipped_existing = 0
    skipped_middle = 0
    review_entries: List[dict] = []

    for img_path, rel, split in iter_unlabeled_images(cfg):
        total += 1
        dest_pose_img = cfg.seed_dataset / "images" / split / rel
        dest_pose_lbl = cfg.seed_dataset / "labels" / split / rel.with_suffix(".txt")

        dest_review_img = cfg.review_queue / "images" / split / rel
        dest_review_lbl = cfg.review_queue / "labels" / split / rel.with_suffix(".txt")

        pose_has_label = label_has_content(dest_pose_lbl)
        review_has_label = label_has_content(dest_review_lbl)
        if not cfg.overwrite_existing and (pose_has_label or review_has_label):
            skipped_existing += 1
            continue

        result = model.predict(source=str(img_path), conf=cfg.predict_conf, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0 or getattr(result, "keypoints", None) is None:
            # No detection - check if this is a negative sample
            if "non" in img_path.name.lower():
                # Negative sample (no lizard) -> send to pose-seed with empty label
                target_img, target_lbl = dest_pose_img, dest_pose_lbl
                target = "pose-seed (negative sample)"
                label_line = ""
                conf_class = "negative"
                meta_conf = {"box": 0.0, "kpts": []}
            else:
                # No detection on regular image -> send to review queue
                target_img, target_lbl = dest_review_img, dest_review_lbl
                target = "review (no detection)"
                label_line = ""
                conf_class = "low"
                meta_conf = {"box": 0.0, "kpts": []}
        else:
            det_idx = int(result.boxes.conf.argmax().item())
            label_line = detection_to_label_line(result, det_idx, cfg) or ""
            conf_class, box_conf, kpt_scores = classify_confidence(result, det_idx, cfg)
            meta_conf = {"box": box_conf, "kpts": kpt_scores}
            
            if conf_class == "high":
                target_img, target_lbl = dest_pose_img, dest_pose_lbl
                target = "pose-seed (high confidence)"
            elif conf_class == "low":
                target_img, target_lbl = dest_review_img, dest_review_lbl
                target = "review (low confidence)"
            else:  # conf_class == "skip"
                # Skip this sample - don't route anywhere
                print(f"[infer] {img_path} -> SKIPPED (middle confidence: box={box_conf:.3f}, kpts={kpt_scores})")
                skipped_middle += 1
                continue

        target_img.parent.mkdir(parents=True, exist_ok=True)
        target_lbl.parent.mkdir(parents=True, exist_ok=True)

        if not cfg.dry_run:
            shutil.copy2(img_path, target_img)
            if label_line:
                target_lbl.write_text(label_line + "\n", encoding="utf-8")
            else:
                target_lbl.write_text("", encoding="utf-8")

        if "pose-seed" in target:
            routed_to_pose_seed += 1
        elif "review" in target:
            routed_to_review += 1
            review_entries.append(
                {
                    "source": str(img_path),
                    "image": str(target_img),
                    "label": str(target_lbl),
                    "box_conf": meta_conf["box"],
                    "kpt_conf": meta_conf["kpts"],
                    "reason": target,
                }
            )

        print(f"[infer] {img_path} -> {target} (box={meta_conf['box']:.3f}, kpts={meta_conf['kpts']})")

    if review_entries:
        # Append to queue manifest so labelers can pick them up.
        with cfg.queue_manifest.open("a", encoding="utf-8") as f:
            for entry in review_entries:
                f.write(json.dumps(entry) + "\n")

    print(
        f"[infer] processed={total}, sent_to_pose_seed={routed_to_pose_seed}, "
        f"sent_to_review={routed_to_review}, skipped_middle_conf={skipped_middle}, skipped_existing={skipped_existing}"
    )


def main() -> None:
    cfg = CONFIG
    ensure_structure(cfg)
    # Optionally trim the seed dataset to a fixed size before training.
    trim_pose_seed(cfg)
    weights_path = train_pose_model(cfg)
    if weights_path:
        route_predictions(cfg, weights_path)
    else:
        print("[warn] no weights produced; inference skipped.")


if __name__ == "__main__":
    main()
