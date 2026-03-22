# Zero-Shot AEC Object Detection

Zero-shot object detection for construction site audio equipment using **OWL-ViTv2** (`google/owlv2-base-patch16-ensemble`). No training data required — detects objects from plain-English text descriptions.

**GitHub:** [JChiaHH/Object\_Detection\_Adaptation](https://github.com/JChiaHH/Object_Detection_Adaptation/tree/main)

## Detected Classes

| Class | Description |
|---|---|
| `portable_sound_meter` | Handheld digital sound level meter held by a person |
| `fixed_noise_monitor` | Weatherproof enclosure on tripod or pole |
| `portable_sound_barrier` | Freestanding modular panels or inflatable enclosures around machinery |
| `fixed_sound_barrier` | Large permanent hoarding wall at construction site perimeter |
| `measuring_tape` | Retractable tape measure cassette with extended blade |

---

## Setup

```bash
conda create -n aec-detection python=3.11 -y
conda activate aec-detection
pip install torch torchvision transformers Pillow numpy
```

---

## Project Structure

```
├── frames/                     # Input images (JPEG/PNG)
├── output/
│   ├── detections.json         # Raw detection results
│   ├── evaluation.json         # mAP evaluation results
│   └── visualizations/         # Annotated output images
├── annotations/                # LabelMe JSON annotations (ground truth)
├── ground_truth.json           # Compiled ground truth bounding boxes
├── inference.py                # Main detection script
├── evaluate.py                 # mAP evaluation script
├── detections_to_labelme.py    # Convert detections → LabelMe pre-annotations
└── convert_annotations.py      # Convert LabelMe/XML annotations → ground_truth.json
```

---

## Usage

### 1. Run Inference

Place your images in `./frames/` then run:

```bash
python inference.py --frames_dir ./frames --output_dir ./output
```

Outputs:
- `output/detections.json` — bounding boxes with confidence scores
- `output/visualizations/det_*.jpg` — visualised detections overlaid on frames

### 2. Evaluate (mAP)

Requires `ground_truth.json` (see Annotation section below).

```bash
python evaluate.py \
    --detections ./output/detections.json \
    --ground_truth ./ground_truth.json \
    --output ./output/evaluation.json
```

### 3. Annotate Ground Truth (optional)

**Step 1** — Convert detections to LabelMe pre-annotations:
```bash
python detections_to_labelme.py \
    --detections ./output/detections.json \
    --frames_dir ./frames \
    --output_dir ./annotations
```

**Step 2** — Open LabelMe to review/correct:
```bash
python -m labelme
```

**Step 3** — Compile annotations to `ground_truth.json`:
```bash
python convert_annotations.py \
    --ann_dir ./annotations \
    --format labelme \
    --output ./ground_truth.json
```

---

## Results

Evaluated on 20 manually annotated egocentric construction site frames using the COCO mAP protocol.

| Class | N GT | AP@0.5 | AP@0.5:0.95 |
|---|---|---|---|
| Portable sound meter | 3 | 0.6634 | 0.6634 |
| Fixed noise monitor | 4 | 0.5644 | 0.5644 |
| Portable sound barrier | 5 | 0.5474 | 0.5474 |
| Fixed sound barrier | 4 | **1.0000** | **1.0000** |
| Measuring tape | 8 | 0.8713 | 0.7753 |
| **mAP (mean)** | | **0.7293** | **0.7101** |

### Key design decisions

- **Prompt ensembling** — 3 text variants per class (15 prompts total) in a single forward pass; max score per class per patch is retained. Covers intra-class visual diversity (e.g., blue inflatable vs. grey quilted barriers).
- **Per-class confidence thresholds** — tuned from TP/FP confidence distributions rather than a single global threshold.
- **Cross-class NMS** — suppresses the lower-confidence box when two different classes overlap the same region.
- **Proximity-based box merging** — merges same-class boxes whose gap is < 50% of the shorter box side (handles split tape measure detections).

---

## Model

- **Model:** `google/owlv2-base-patch16-ensemble`
- **Speed:** ~3.7 FPS on NVIDIA RTX 5070 Ti Laptop GPU
- **Token limit:** OWL-ViTv2's text encoder has a hard 16-token limit (including special tokens). All prompts are kept within this budget.
