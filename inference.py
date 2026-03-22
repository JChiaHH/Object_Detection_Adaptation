"""
Task A: Object Detection Adaptation — OWL-ViTv2 Inference Script
================================================================
Detects sound level meters, acoustic panels, and measuring tapes
in egocentric construction site frames using zero-shot OWL-ViTv2.

Usage:
    python inference.py --frames_dir ./frames --output_dir ./output

Requirements:
    pip install torch torchvision transformers Pillow numpy
    pip install pycocotools  # for mAP evaluation
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import Owlv2Processor, Owlv2ForObjectDetection


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
DEFAULT_MODEL = "google/owlv2-base-patch16-ensemble"
DEFAULT_CONFIDENCE = 0.30  # fallback; per-class thresholds below take priority
DEFAULT_NMS_IOU = 0.3

# Per-class confidence thresholds (tuned from confidence distribution analysis)
# measuring_tape: clean class, no FPs → 0.35
# barriers: FPs cluster 0.32–0.34, TPs cluster 0.40+ → 0.40
# sound meters: overlapping TP/FP scores → 0.45 (accept some FNs)
CLASS_THRESHOLDS = {
    "portable_sound_meter": 0.40,
    "fixed_noise_monitor":  0.40,
    "portable_sound_barrier": 0.35,
    "fixed_sound_barrier":  0.37,
    "measuring_tape":       0.33,
}

# Primary text prompts — kept within 16-token OWLv2 hard limit
TEXT_PROMPTS = [
    "handheld sound level meter held by person",
    "noise monitoring box mounted on a pole",
    "portable acoustic barrier or enclosure around construction equipment",
    "tall hoarding wall along construction site boundary",
    "tape measure with measurement markings",
]

# Class name mapping (prompt index → label)
CLASS_NAMES = {
    0: "portable_sound_meter",
    1: "fixed_noise_monitor",
    2: "portable_sound_barrier",
    3: "fixed_sound_barrier",
    4: "measuring_tape",
}

# Alternative prompts for sensitivity ablation / ensembling
ABLATION_PROMPTS = {
    "portable_sound_meter": [
        "handheld sound level meter held by person",
        "person holding a decibel meter",
        "sound level meter with microphone",
    ],
    "fixed_noise_monitor": [
        "noise monitoring box mounted on a pole",
        "weatherproof instrument box on tripod outdoors",
        "environmental noise logger on a pole",
    ],
    "portable_sound_barrier": [
        "portable acoustic barrier enclosure around construction equipment",
        "acoustic blanket or quilted panel barrier around machinery",
        "portable noise enclosure panels around construction machine",
    ],
    "fixed_sound_barrier": [
        "tall hoarding wall along construction site boundary",
        "large corrugated metal noise barrier surrounding building site",
        "continuous rigid noise barrier fence at construction perimeter",
    ],
    "measuring_tape": [
        "yellow tape measure cassette",
        "retractable tape measure with extended blade",
        "tape measure held in hand on construction site",
    ],
}

# Visualization colors per class (RGB)
CLASS_COLORS = {
    "portable_sound_meter": (46, 107, 158),
    "fixed_noise_monitor": (90, 150, 200),
    "portable_sound_barrier": (30, 158, 117),
    "fixed_sound_barrier": (20, 100, 80),
    "measuring_tape": (216, 90, 48),
}


# ─────────────────────────────────────────────
# NMS
# ─────────────────────────────────────────────
def nms_per_class(boxes, scores, labels, iou_threshold=0.5):
    """Per-class greedy Non-Maximum Suppression."""
    keep = []
    unique_labels = set(labels)
    for cls in unique_labels:
        cls_mask = [i for i, l in enumerate(labels) if l == cls]
        if not cls_mask:
            continue
        cls_boxes = torch.tensor([boxes[i] for i in cls_mask], dtype=torch.float32)
        cls_scores = torch.tensor([scores[i] for i in cls_mask], dtype=torch.float32)

        # Sort by score descending
        order = cls_scores.argsort(descending=True)
        suppressed = set()
        for i in range(len(order)):
            idx_i = order[i].item()
            if idx_i in suppressed:
                continue
            keep.append(cls_mask[idx_i])
            for j in range(i + 1, len(order)):
                idx_j = order[j].item()
                if idx_j in suppressed:
                    continue
                iou = compute_iou(cls_boxes[idx_i], cls_boxes[idx_j])
                if iou > iou_threshold:
                    suppressed.add(idx_j)
    return keep


def cross_class_nms(detections: list[dict], iou_threshold: float = 0.3) -> list[dict]:
    """
    Suppress lower-confidence boxes that overlap significantly with a
    higher-confidence box of a different class. Keeps the best-scoring
    box when two classes fire on the same region.
    """
    detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    suppressed = [False] * len(detections)
    for i in range(len(detections)):
        if suppressed[i]:
            continue
        for j in range(i + 1, len(detections)):
            if suppressed[j]:
                continue
            if detections[i]["class_name"] == detections[j]["class_name"]:
                continue  # same-class handled by per-class NMS already
            iou = compute_iou(detections[i]["bbox_pixel"], detections[j]["bbox_pixel"])
            if iou > iou_threshold:
                suppressed[j] = True
    return [d for d, s in zip(detections, suppressed) if not s]


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


def merge_nearby_boxes(detections: list[dict], gap_ratio: float = 0.5) -> list[dict]:
    """
    Merge same-class boxes that are close to each other into their union.
    Two boxes are merged if the gap between them is less than gap_ratio * the
    shorter side of the smaller box. Repeats until no merges remain.
    """
    def gap_between(b1, b2):
        """Minimum pixel gap between two [x1,y1,x2,y2] boxes (0 if overlapping)."""
        dx = max(0, max(b1[0], b2[0]) - min(b1[2], b2[2]))
        dy = max(0, max(b1[1], b2[1]) - min(b1[3], b2[3]))
        return max(dx, dy)

    def shorter_side(b):
        return min(b[2] - b[0], b[3] - b[1])

    def union_box(b1, b2):
        return [min(b1[0], b2[0]), min(b1[1], b2[1]),
                max(b1[2], b2[2]), max(b1[3], b2[3])]

    changed = True
    while changed:
        changed = False
        merged = []
        used = [False] * len(detections)
        for i in range(len(detections)):
            if used[i]:
                continue
            det_i = detections[i]
            for j in range(i + 1, len(detections)):
                if used[j]:
                    continue
                det_j = detections[j]
                if det_i["class_name"] != det_j["class_name"]:
                    continue
                threshold = gap_ratio * min(shorter_side(det_i["bbox_pixel"]),
                                            shorter_side(det_j["bbox_pixel"]))
                if gap_between(det_i["bbox_pixel"], det_j["bbox_pixel"]) <= threshold:
                    new_box = union_box(det_i["bbox_pixel"], det_j["bbox_pixel"])
                    det_i = {**det_i,
                             "bbox_pixel": new_box,
                             "confidence": max(det_i["confidence"], det_j["confidence"])}
                    used[j] = True
                    changed = True
            merged.append(det_i)
            used[i] = True
        detections = merged
    return detections


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────
class OWLv2Detector:
    def __init__(self, model_name=DEFAULT_MODEL, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {model_name} on {self.device}...")
        self.processor = Owlv2Processor.from_pretrained(model_name)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("Model loaded.")

    def _build_ensemble_prompts(self, text_queries: list[str]) -> tuple[list[str], dict]:
        """
        Expand primary prompts with ABLATION_PROMPTS variants.
        Returns (flat prompt list, mapping from flat index → class_id).
        """
        flat_prompts = []
        idx_to_class = {}
        for cls_id, primary in enumerate(text_queries):
            cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
            variants = ABLATION_PROMPTS.get(cls_name, [primary])
            # Always include the primary prompt; deduplicate if already in variants
            all_variants = list(dict.fromkeys([primary] + variants))
            for prompt in all_variants:
                idx_to_class[len(flat_prompts)] = cls_id
                flat_prompts.append(prompt)
        return flat_prompts, idx_to_class

    @torch.no_grad()
    def detect(self, image: Image.Image, text_queries: list[str],
               confidence_threshold: float = DEFAULT_CONFIDENCE,
               nms_iou: float = DEFAULT_NMS_IOU) -> list[dict]:
        """
        Run detection on a single image using prompt ensembling.
        All ABLATION_PROMPTS variants are passed simultaneously; the highest
        score per class per box is kept, then standard NMS is applied.

        Returns list of dicts:
            {class_id, class_name, confidence, bbox_pixel, bbox_normalized}
        """
        w, h = image.size

        # Build ensemble: expand to all prompt variants across all classes
        flat_prompts, idx_to_class = self._build_ensemble_prompts(text_queries)

        inputs = self.processor(
            text=[flat_prompts],
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        outputs = self.model(**inputs)

        # Post-process at very low threshold — we filter per class below
        min_threshold = min(CLASS_THRESHOLDS.values(), default=confidence_threshold)
        target_sizes = torch.tensor([[h, w]], device=self.device)
        results = self.processor.image_processor.post_process_object_detection(
            outputs, threshold=min_threshold * 0.5, target_sizes=target_sizes
        )[0]

        raw_boxes = results["boxes"].cpu().tolist()
        raw_scores = results["scores"].cpu().tolist()
        raw_labels = results["labels"].cpu().tolist()

        if not raw_boxes:
            return []

        # Re-map prompt indices → class ids; keep best score per (box, class) pair.
        # Group by box position (rounded to reduce floating-point jitter).
        box_class_best: dict[tuple, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        for box, score, prompt_idx in zip(raw_boxes, raw_scores, raw_labels):
            cls_id = idx_to_class.get(prompt_idx, prompt_idx)
            box_key = tuple(round(v, 0) for v in box)
            if score > box_class_best[box_key][cls_id]:
                box_class_best[box_key][cls_id] = score

        # Flatten back to box/score/label lists using best score per class
        boxes, scores, labels = [], [], []
        for box_key, cls_scores in box_class_best.items():
            for cls_id, score in cls_scores.items():
                boxes.append(list(box_key))
                scores.append(score)
                labels.append(cls_id)

        if not boxes:
            return []

        # Per-class NMS
        keep_indices = nms_per_class(boxes, scores, labels, iou_threshold=nms_iou)

        detections = []
        for idx in keep_indices:
            x1, y1, x2, y2 = boxes[idx]
            cls_id = labels[idx]
            cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
            cls_threshold = CLASS_THRESHOLDS.get(cls_name, confidence_threshold)
            if scores[idx] < cls_threshold:
                continue
            detections.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": round(scores[idx], 4),
                "bbox_pixel": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "bbox_normalized": [
                    round(x1 / w, 6), round(y1 / h, 6),
                    round((x2 - x1) / w, 6), round((y2 - y1) / h, 6),
                ],  # COCO-style: [x, y, width, height] normalized
            })

        # Cross-class NMS: suppress lower-confidence box when two classes overlap
        detections = cross_class_nms(detections, iou_threshold=nms_iou)

        # Merge nearby same-class boxes (e.g. tape cassette + extended blade)
        detections = merge_nearby_boxes(detections)

        # Recompute normalized coords after merging
        for det in detections:
            x1, y1, x2, y2 = det["bbox_pixel"]
            det["bbox_normalized"] = [
                round(x1 / w, 6), round(y1 / h, 6),
                round((x2 - x1) / w, 6), round((y2 - y1) / h, 6),
            ]

        # Sort by confidence descending
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────
def draw_detections(image: Image.Image, detections: list[dict],
                    output_path: str) -> None:
    """Draw bounding boxes and labels on image and save."""
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
        font_small = font

    for det in detections:
        x1, y1, x2, y2 = det["bbox_pixel"]
        color = CLASS_COLORS.get(det["class_name"], (200, 200, 200))
        label = f'{det["class_name"]} {det["confidence"]:.2f}'

        # Box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Label background
        bbox = font_small.getbbox(label)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([x1, y1 - th - 6, x1 + tw + 8, y1], fill=color)
        draw.text((x1 + 4, y1 - th - 4), label, fill=(255, 255, 255), font=font_small)

    image.save(output_path)


# ─────────────────────────────────────────────
# Evaluation (COCO-style mAP)
# ─────────────────────────────────────────────
def compute_ap(precisions, recalls):
    """101-point interpolated AP (COCO standard)."""
    recall_thresholds = np.linspace(0, 1, 101)
    interp_precisions = np.zeros_like(recall_thresholds)

    for i, r_thresh in enumerate(recall_thresholds):
        precisions_above = [p for p, r in zip(precisions, recalls) if r >= r_thresh]
        interp_precisions[i] = max(precisions_above) if precisions_above else 0.0

    return np.mean(interp_precisions)


def evaluate_detections(all_detections: dict, ground_truth_path: str,
                        iou_thresholds: list[float] = None) -> dict:
    """
    Compute per-class AP at specified IoU thresholds.

    Args:
        all_detections: {frame_name: [detection_dicts]}
        ground_truth_path: path to ground_truth.json
        iou_thresholds: list of IoU thresholds (default: 0.5 to 0.95 step 0.05)

    Returns:
        Dict with per-class and mean AP at each threshold.
    """
    if iou_thresholds is None:
        iou_thresholds = [round(0.5 + i * 0.05, 2) for i in range(10)]

    if not os.path.exists(ground_truth_path):
        print(f"[WARN] Ground truth file not found: {ground_truth_path}")
        print("       Skipping evaluation. Create ground_truth.json to enable mAP computation.")
        return {}

    with open(ground_truth_path) as f:
        gt_data = json.load(f)

    results = {}

    for iou_thresh in iou_thresholds:
        per_class_ap = {}

        for cls_name in CLASS_NAMES.values():
            # Collect all predictions and GT for this class
            all_preds = []
            total_gt = 0

            for frame_name, gt_boxes in gt_data.items():
                gt_cls = [b for b in gt_boxes if b["class_name"] == cls_name]
                total_gt += len(gt_cls)

                preds = all_detections.get(frame_name, [])
                pred_cls = [p for p in preds if p["class_name"] == cls_name]
                pred_cls = sorted(pred_cls, key=lambda p: p["confidence"], reverse=True)

                gt_matched = [False] * len(gt_cls)

                for pred in pred_cls:
                    best_iou = 0
                    best_gt_idx = -1
                    for gt_idx, gt_box in enumerate(gt_cls):
                        if gt_matched[gt_idx]:
                            continue
                        iou = compute_iou(pred["bbox_pixel"], gt_box["bbox_pixel"])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx

                    if best_iou >= iou_thresh and best_gt_idx >= 0 and not gt_matched[best_gt_idx]:
                        all_preds.append((pred["confidence"], True))
                        gt_matched[best_gt_idx] = True
                    else:
                        all_preds.append((pred["confidence"], False))

            if total_gt == 0:
                per_class_ap[cls_name] = 0.0
                continue

            # Sort by confidence descending
            all_preds.sort(key=lambda x: x[0], reverse=True)

            tp_cumsum = 0
            fp_cumsum = 0
            precisions = []
            recalls = []

            for conf, is_tp in all_preds:
                if is_tp:
                    tp_cumsum += 1
                else:
                    fp_cumsum += 1
                precisions.append(tp_cumsum / (tp_cumsum + fp_cumsum))
                recalls.append(tp_cumsum / total_gt)

            per_class_ap[cls_name] = compute_ap(precisions, recalls)

        mean_ap = np.mean(list(per_class_ap.values()))
        results[f"IoU_{iou_thresh:.2f}"] = {
            "per_class": {k: round(v, 4) for k, v in per_class_ap.items()},
            "mAP": round(mean_ap, 4),
        }

    return results


# ─────────────────────────────────────────────
# F1 sweep for threshold selection
# ─────────────────────────────────────────────
def sweep_thresholds(detector, images, ground_truth_path,
                     thresholds=None) -> dict:
    """
    Sweep confidence thresholds and report F1 for each.
    Used to justify the chosen threshold.
    """
    if thresholds is None:
        thresholds = [round(0.05 + i * 0.05, 2) for i in range(10)]

    if not os.path.exists(ground_truth_path):
        print("[WARN] Ground truth not found, skipping threshold sweep.")
        return {}

    with open(ground_truth_path) as f:
        gt_data = json.load(f)

    results = {}
    for thresh in thresholds:
        tp, fp, fn = 0, 0, 0
        for frame_name, image in images.items():
            dets = detector.detect(image, TEXT_PROMPTS,
                                   confidence_threshold=thresh)
            gt_boxes = gt_data.get(frame_name, [])
            gt_matched = [False] * len(gt_boxes)

            for det in dets:
                matched = False
                for gi, gb in enumerate(gt_boxes):
                    if gb["class_name"] == det["class_name"] and not gt_matched[gi]:
                        iou = compute_iou(det["bbox_pixel"], gb["bbox_pixel"])
                        if iou >= 0.5:
                            tp += 1
                            gt_matched[gi] = True
                            matched = True
                            break
                if not matched:
                    fp += 1
            fn += sum(1 for m in gt_matched if not m)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        results[thresh] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    return results


# ─────────────────────────────────────────────
# Prompt ablation
# ─────────────────────────────────────────────
def run_prompt_ablation(detector, images, ground_truth_path) -> dict:
    """
    For each class, test alternative prompts and report AP@0.5.
    """
    if not os.path.exists(ground_truth_path):
        print("[WARN] Ground truth not found, skipping prompt ablation.")
        return {}

    results = {}
    for cls_name, prompts in ABLATION_PROMPTS.items():
        results[cls_name] = {}
        cls_id = [k for k, v in CLASS_NAMES.items() if v == cls_name][0]

        for prompt in prompts:
            # Build full prompt list: replace the relevant class prompt
            test_prompts = list(TEXT_PROMPTS)
            test_prompts[cls_id] = prompt

            all_dets = {}
            for frame_name, image in images.items():
                dets = detector.detect(image, test_prompts)
                all_dets[frame_name] = dets

            eval_result = evaluate_detections(all_dets, ground_truth_path,
                                              iou_thresholds=[0.5])
            ap = eval_result.get("IoU_0.50", {}).get("per_class", {}).get(cls_name, 0.0)
            results[cls_name][prompt] = round(ap, 4)

    return results


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="OWL-ViTv2 zero-shot object detection for AEC tools")
    parser.add_argument("--frames_dir", type=str, default="./frames",
                        help="Directory containing input frames (JPEG/PNG)")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory for output JSON and visualizations")
    parser.add_argument("--ground_truth", type=str, default="./ground_truth.json",
                        help="Path to ground truth annotations (optional)")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE,
                        help=f"Confidence threshold (default: {DEFAULT_CONFIDENCE})")
    parser.add_argument("--nms_iou", type=float, default=DEFAULT_NMS_IOU,
                        help=f"NMS IoU threshold (default: {DEFAULT_NMS_IOU})")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"HuggingFace model ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip evaluation even if ground truth exists")
    parser.add_argument("--run_ablation", action="store_true",
                        help="Run prompt sensitivity ablation")
    parser.add_argument("--sweep_thresholds", action="store_true",
                        help="Run confidence threshold sweep")
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "visualizations").mkdir(exist_ok=True)

    # Load frames
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    frame_paths = sorted(
        [p for p in frames_dir.iterdir() if p.suffix.lower() in extensions]
    )

    if not frame_paths:
        print(f"No image files found in {frames_dir}")
        return

    print(f"Found {len(frame_paths)} frames in {frames_dir}")

    # Load model
    detector = OWLv2Detector(model_name=args.model)

    # Run inference on all frames
    all_detections = {}
    images = {}
    total_time = 0

    for fp in frame_paths:
        image = Image.open(fp).convert("RGB")
        images[fp.name] = image

        t0 = time.time()
        detections = detector.detect(
            image, TEXT_PROMPTS,
            confidence_threshold=args.confidence,
            nms_iou=args.nms_iou,
        )
        elapsed = time.time() - t0
        total_time += elapsed

        all_detections[fp.name] = detections
        n_det = len(detections)
        print(f"  {fp.name}: {n_det} detection(s) [{elapsed:.2f}s]")

        # Save visualization
        vis_image = image.copy()
        draw_detections(vis_image, detections,
                        str(output_dir / "visualizations" / f"det_{fp.stem}.jpg"))

    # Summary
    avg_fps = len(frame_paths) / total_time if total_time > 0 else 0
    print(f"\nInference complete: {len(frame_paths)} frames, "
          f"{total_time:.1f}s total, {avg_fps:.1f} FPS")

    # Save all detections as JSON
    det_output = {
        "model": args.model,
        "confidence_threshold": args.confidence,
        "nms_iou_threshold": args.nms_iou,
        "text_prompts": TEXT_PROMPTS,
        "num_frames": len(frame_paths),
        "total_inference_time_s": round(total_time, 2),
        "avg_fps": round(avg_fps, 2),
        "detections": all_detections,
    }
    det_path = output_dir / "detections.json"
    with open(det_path, "w") as f:
        json.dump(det_output, f, indent=2)
    print(f"Detections saved to {det_path}")

    # ── Evaluation ──
    if not args.skip_eval and os.path.exists(args.ground_truth):
        print("\n── Evaluation ──")
        eval_results = evaluate_detections(all_detections, args.ground_truth)

        if eval_results:
            # Print headline metrics
            ap50 = eval_results.get("IoU_0.50", {})
            ap50_95 = {
                "per_class": {},
                "mAP": 0.0,
            }
            # Average across all IoU thresholds for mAP@0.5:0.95
            for cls_name in CLASS_NAMES.values():
                vals = [
                    eval_results[k]["per_class"].get(cls_name, 0.0)
                    for k in eval_results
                ]
                ap50_95["per_class"][cls_name] = round(np.mean(vals), 4)
            ap50_95["mAP"] = round(np.mean(list(ap50_95["per_class"].values())), 4)

            print(f"\n{'Class':<22} {'AP@0.5':>8} {'AP@0.5:0.95':>12}")
            print("-" * 44)
            for cls_name in CLASS_NAMES.values():
                a50 = ap50.get("per_class", {}).get(cls_name, 0.0)
                a50_95 = ap50_95["per_class"].get(cls_name, 0.0)
                print(f"{cls_name:<22} {a50:>8.4f} {a50_95:>12.4f}")
            print("-" * 44)
            print(f"{'mAP (mean)':<22} {ap50.get('mAP', 0.0):>8.4f} "
                  f"{ap50_95['mAP']:>12.4f}")

            eval_path = output_dir / "evaluation.json"
            with open(eval_path, "w") as f:
                json.dump({
                    "ap_at_0.50": ap50,
                    "ap_at_0.50_0.95": ap50_95,
                    "full_results": eval_results,
                }, f, indent=2)
            print(f"\nEvaluation saved to {eval_path}")

    # ── Threshold sweep ──
    if args.sweep_thresholds:
        print("\n── Threshold sweep ──")
        sweep = sweep_thresholds(detector, images, args.ground_truth)
        if sweep:
            print(f"\n{'Thresh':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}")
            print("-" * 36)
            for thresh, metrics in sorted(sweep.items()):
                print(f"{thresh:>8.2f} {metrics['precision']:>8.4f} "
                      f"{metrics['recall']:>8.4f} {metrics['f1']:>8.4f}")
            sweep_path = output_dir / "threshold_sweep.json"
            with open(sweep_path, "w") as f:
                json.dump(sweep, f, indent=2)
            print(f"\nSweep saved to {sweep_path}")

    # ── Prompt ablation ──
    if args.run_ablation:
        print("\n── Prompt ablation ──")
        ablation = run_prompt_ablation(detector, images, args.ground_truth)
        if ablation:
            for cls_name, prompts in ablation.items():
                print(f"\n  {cls_name}:")
                for prompt, ap in prompts.items():
                    print(f"    {prompt:<35} AP@0.5 = {ap:.4f}")
            ablation_path = output_dir / "prompt_ablation.json"
            with open(ablation_path, "w") as f:
                json.dump(ablation, f, indent=2)
            print(f"\nAblation saved to {ablation_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
