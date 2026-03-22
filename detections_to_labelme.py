"""
Convert detections.json → labelme JSON pre-annotations
=======================================================
Usage:
    python detections_to_labelme.py \
        --detections ./output/detections.json \
        --frames_dir ./frames \
        --output_dir ./annotations
"""

import argparse
import base64
import json
from pathlib import Path
from PIL import Image
import io


def image_to_b64(img_path: Path) -> str:
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", default="./output/detections.json")
    parser.add_argument("--frames_dir", default="./frames")
    parser.add_argument("--output_dir", default="./annotations")
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.detections) as f:
        data = json.load(f)
    detections = data.get("detections", data)

    written = 0
    for frame_name, boxes in detections.items():
        img_path = frames_dir / frame_name
        if not img_path.exists():
            matches = list(frames_dir.glob(Path(frame_name).stem + ".*"))
            if not matches:
                print(f"  [SKIP] Image not found: {frame_name}")
                continue
            img_path = matches[0]

        img = Image.open(img_path)
        w, h = img.size

        shapes = []
        for det in boxes:
            x1, y1, x2, y2 = det["bbox_pixel"]
            shapes.append({
                "label": det["class_name"],
                "points": [[x1, y1], [x2, y2]],
                "group_id": None,
                "description": f"confidence: {det['confidence']}",
                "shape_type": "rectangle",
                "flags": {},
            })

        labelme_json = {
            "version": "5.3.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": str(img_path.resolve()),
            "imageData": None,   # don't embed image data — keeps files small
            "imageHeight": h,
            "imageWidth": w,
        }

        out_path = output_dir / (img_path.stem + ".json")
        with open(out_path, "w") as f:
            json.dump(labelme_json, f, indent=2)

        status = f"{len(boxes)} box(es)" if boxes else "no detections — draw manually"
        print(f"  {img_path.name} → {out_path.name}  [{status}]")
        written += 1

    print(f"\nWrote {written} JSON files to {output_dir}")


if __name__ == "__main__":
    main()
