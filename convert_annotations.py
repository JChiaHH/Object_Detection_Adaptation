"""
Convert annotations → ground_truth.json
Supports labelme JSON and LabelImg PASCAL VOC XML.

Usage:
    # labelme (default)
    python convert_annotations.py --ann_dir ./annotations --output ./ground_truth.json

    # LabelImg XML
    python convert_annotations.py --ann_dir ./frames --format xml --output ./ground_truth.json
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

CLASS_MAP = {
    "sound_level_meter":      "portable_sound_meter",
    "noise_monitor":          "fixed_noise_monitor",
    "acoustic_panel":         "portable_sound_barrier",
    "sound_barrier":          "fixed_sound_barrier",
    "measuring_tape":         "measuring_tape",
    "portable_sound_meter":   "portable_sound_meter",
    "fixed_noise_monitor":    "fixed_noise_monitor",
    "portable_sound_barrier": "portable_sound_barrier",
    "fixed_sound_barrier":    "fixed_sound_barrier",
}


def parse_labelme(json_path: Path) -> tuple[str, list[dict]]:
    with open(json_path) as f:
        data = json.load(f)
    filename = Path(data.get("imagePath", json_path.stem)).name
    annotations = []
    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "rectangle":
            continue
        raw_class = shape.get("label", "").strip()
        cls_name = CLASS_MAP.get(raw_class, raw_class)
        pts = shape["points"]  # [[x1,y1],[x2,y2]]
        x1, y1 = pts[0]
        x2, y2 = pts[1]
        annotations.append({"class_name": cls_name,
                             "bbox_pixel": [float(x1), float(y1), float(x2), float(y2)]})
    return filename, annotations


def parse_xml(xml_path: Path) -> tuple[str, list[dict]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename = root.findtext("filename") or xml_path.stem
    annotations = []
    for obj in root.findall("object"):
        raw_class = obj.findtext("name", "").strip()
        cls_name = CLASS_MAP.get(raw_class, raw_class)
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        x1 = float(bndbox.findtext("xmin", 0))
        y1 = float(bndbox.findtext("ymin", 0))
        x2 = float(bndbox.findtext("xmax", 0))
        y2 = float(bndbox.findtext("ymax", 0))
        annotations.append({"class_name": cls_name, "bbox_pixel": [x1, y1, x2, y2]})
    return filename, annotations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_dir", default="./annotations")
    parser.add_argument("--format", choices=["labelme", "xml"], default="labelme")
    parser.add_argument("--output", default="./ground_truth.json")
    args = parser.parse_args()

    ann_dir = Path(args.ann_dir)

    if args.format == "labelme":
        files = sorted(ann_dir.glob("*.json"))
        parse_fn = parse_labelme
    else:
        files = sorted(ann_dir.glob("*.xml"))
        parse_fn = parse_xml

    if not files:
        print(f"No {args.format} files found in {ann_dir}")
        return

    ground_truth = {}
    for ann_path in files:
        filename, annotations = parse_fn(ann_path)
        if annotations:
            ground_truth[filename] = annotations
            print(f"  {filename}: {len(annotations)} box(es)")
        else:
            print(f"  {filename}: (empty — skipped)")

    with open(args.output, "w") as f:
        json.dump(ground_truth, f, indent=2)
    print(f"\nSaved {len(ground_truth)} annotated frames → {args.output}")


if __name__ == "__main__":
    main()
