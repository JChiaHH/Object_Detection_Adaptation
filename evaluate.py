"""
mAP Evaluation Script for OWL-ViTv2 Zero-Shot Detection
=========================================================
Computes per-class AP@0.5 and mAP@[0.5:0.95] from:
  - detections.json   (produced by inference.py)
  - ground_truth.json (manually annotated bounding boxes)

Usage:
    python evaluate.py \
        --detections ./output/detections.json \
        --ground_truth ./ground_truth.json \
        --output ./output/evaluation.json

Ground truth JSON format:
    {
        "frame.jpg": [
            {"class_name": "measuring_tape", "bbox_pixel": [x1, y1, x2, y2]},
            ...
        ],
        ...
    }
"""

import argparse
import json
import numpy as np
from pathlib import Path


# Map original 3-class names → 5-class names if ground truth uses old naming
CLASS_ALIASES = {
    "sound_level_meter": ["portable_sound_meter", "fixed_noise_monitor"],
    "acoustic_panel":    ["portable_sound_barrier", "fixed_sound_barrier"],
    "measuring_tape":    ["measuring_tape"],
}

ALL_CLASSES = [
    "portable_sound_meter",
    "fixed_noise_monitor",
    "portable_sound_barrier",
    "fixed_sound_barrier",
    "measuring_tape",
]


def compute_iou(box1, box2):
    """IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def interpolated_ap(precisions, recalls):
    """101-point interpolated AP (COCO standard)."""
    recall_thresholds = np.linspace(0, 1, 101)
    interp = np.zeros(101)
    for i, r in enumerate(recall_thresholds):
        p_above = [p for p, rc in zip(precisions, recalls) if rc >= r]
        interp[i] = max(p_above) if p_above else 0.0
    return float(np.mean(interp))


def compute_ap_for_class(cls_name, all_detections, gt_data, iou_threshold):
    """
    Compute AP for a single class at a given IoU threshold.
    Handles the case where GT uses 3-class names by matching any alias.
    """
    # Which GT class names map to this detection class
    gt_aliases = {cls_name}
    for old_name, new_names in CLASS_ALIASES.items():
        if cls_name in new_names:
            gt_aliases.add(old_name)

    total_gt = 0
    all_preds = []  # (confidence, is_tp)

    for frame_name in sorted(all_detections.keys()):
        gt_boxes_raw = gt_data.get(frame_name, [])
        gt_cls = [b for b in gt_boxes_raw if b["class_name"] in gt_aliases]
        total_gt += len(gt_cls)

        preds = all_detections.get(frame_name, [])
        pred_cls = [p for p in preds if p["class_name"] == cls_name]
        pred_cls = sorted(pred_cls, key=lambda x: x["confidence"], reverse=True)

        gt_matched = [False] * len(gt_cls)
        for pred in pred_cls:
            best_iou, best_idx = 0.0, -1
            for gi, gb in enumerate(gt_cls):
                if gt_matched[gi]:
                    continue
                iou = compute_iou(pred["bbox_pixel"], gb["bbox_pixel"])
                if iou > best_iou:
                    best_iou, best_idx = iou, gi
            if best_iou >= iou_threshold and best_idx >= 0:
                all_preds.append((pred["confidence"], True))
                gt_matched[best_idx] = True
            else:
                all_preds.append((pred["confidence"], False))

    if total_gt == 0:
        return 0.0, 0, 0  # AP, total_gt, total_pred

    all_preds.sort(key=lambda x: x[0], reverse=True)
    tp_cum, fp_cum = 0, 0
    precisions, recalls = [], []
    for _, is_tp in all_preds:
        if is_tp:
            tp_cum += 1
        else:
            fp_cum += 1
        precisions.append(tp_cum / (tp_cum + fp_cum))
        recalls.append(tp_cum / total_gt)

    return interpolated_ap(precisions, recalls), total_gt, len(all_preds)


def evaluate(detections_path, gt_path, iou_thresholds=None):
    if iou_thresholds is None:
        iou_thresholds = [round(0.5 + i * 0.05, 2) for i in range(10)]

    with open(detections_path) as f:
        det_data = json.load(f)
    with open(gt_path) as f:
        gt_data = json.load(f)

    all_detections = det_data.get("detections", det_data)

    results = {}
    for iou_thresh in iou_thresholds:
        per_class = {}
        for cls_name in ALL_CLASSES:
            ap, n_gt, n_pred = compute_ap_for_class(
                cls_name, all_detections, gt_data, iou_thresh
            )
            per_class[cls_name] = {"ap": round(ap, 4), "n_gt": n_gt, "n_pred": n_pred}
        mean_ap = float(np.mean([v["ap"] for v in per_class.values()]))
        results[f"IoU_{iou_thresh:.2f}"] = {
            "per_class": per_class,
            "mAP": round(mean_ap, 4),
        }

    # Compute AP@0.5 and AP@0.5:0.95 summary
    ap50 = results["IoU_0.50"]
    ap50_95 = {}
    for cls_name in ALL_CLASSES:
        vals = [results[k]["per_class"][cls_name]["ap"] for k in results]
        ap50_95[cls_name] = round(float(np.mean(vals)), 4)
    map50_95 = round(float(np.mean(list(ap50_95.values()))), 4)

    # Collect GT instance counts from IoU_0.50 (same across all thresholds)
    n_gt = {cls_name: ap50["per_class"][cls_name]["n_gt"] for cls_name in ALL_CLASSES}

    summary = {
        "AP@0.50": {
            "per_class": {k: ap50["per_class"][k]["ap"] for k in ALL_CLASSES},
            "mAP": ap50["mAP"],
        },
        "AP@0.50:0.95": {
            "per_class": ap50_95,
            "mAP": map50_95,
        },
        "n_gt": n_gt,
        "full_results": results,
        "metadata": {
            "detections_file": str(detections_path),
            "ground_truth_file": str(gt_path),
            "num_frames": len(all_detections),
            "classes": ALL_CLASSES,
        },
    }
    return summary


def print_table(summary):
    ap50    = summary["AP@0.50"]
    ap50_95 = summary["AP@0.50:0.95"]
    n_gt    = summary.get("n_gt", {})

    print(f"\n{'Class':<26} {'N_GT':>6} {'AP@0.5':>8} {'AP@0.5:0.95':>13}")
    print("─" * 58)
    for cls_name in ALL_CLASSES:
        a50    = ap50["per_class"][cls_name]
        a5095  = ap50_95["per_class"][cls_name]
        n      = n_gt.get(cls_name, "?")
        print(f"{cls_name:<26} {n:>6} {a50:>8.4f} {a5095:>13.4f}")
    print("─" * 58)
    print(f"{'mAP (mean)':<26} {'':>6} {ap50['mAP']:>8.4f} {ap50_95['mAP']:>13.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Compute mAP for OWL-ViTv2 detections")
    parser.add_argument("--detections",   default="./output/detections.json")
    parser.add_argument("--ground_truth", default="./ground_truth.json")
    parser.add_argument("--output",       default="./output/evaluation.json")
    args = parser.parse_args()

    det_path = Path(args.detections)
    gt_path  = Path(args.ground_truth)
    out_path = Path(args.output)

    if not det_path.exists():
        print(f"[ERROR] Detections file not found: {det_path}")
        return
    if not gt_path.exists():
        print(f"[ERROR] Ground truth file not found: {gt_path}")
        print("        Create it using ground_truth_template.json as a guide.")
        return

    print(f"Evaluating: {det_path}")
    print(f"Against:    {gt_path}")

    summary = evaluate(det_path, gt_path)
    print_table(summary)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
