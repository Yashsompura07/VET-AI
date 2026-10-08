"""Shared helpers to turn a Roboflow COCO export into a disease-classification dataset.

The Roboflow "Cattle Disease Detection" export has bounding-box labels with classes
like: Infected_Foot_Image, Mouth Disease Infected, Normal_Healthy_Cow,
Normal_Mouth_Image, lumpy skin. We convert each IMAGE to a single disease label so we
can train an image classifier that lines up with the symptom model's diseases.
"""

import json
import os
from collections import Counter

IMG_SIZE = 224

# Our target classes for the app (a subset of what public image data supports).
DISEASE_CLASSES = ["healthy", "fmd", "lsd"]


def map_category_to_disease(name: str):
    """Map a raw Roboflow class name to one of our disease classes (or None to skip)."""
    n = name.lower()
    if "normal" in n or "healthy" in n:
        return "healthy"
    if "lumpy" in n:
        return "lsd"
    if "foot" in n or "mouth" in n:
        return "fmd"   # foot-and-mouth disease
    return None  # unknown class -> ignored


def find_splits(dataset_dir: str):
    """Return {split_name: folder_path} for any of train/valid/test that exist."""
    splits = {}
    for split in ["train", "valid", "test"]:
        p = os.path.join(dataset_dir, split)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "_annotations.coco.json")):
            splits[split] = p
    # Some exports put everything at the top level
    if not splits and os.path.exists(os.path.join(dataset_dir, "_annotations.coco.json")):
        splits["train"] = dataset_dir
    return splits


def load_split(split_dir: str):
    """Return (filepaths, labels) for one split, using the majority category per image."""
    with open(os.path.join(split_dir, "_annotations.coco.json"), encoding="utf-8") as f:
        coco = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    img_id_to_file = {im["id"]: im["file_name"] for im in coco["images"]}

    # collect all category ids seen per image
    per_image_cats = {}
    for ann in coco["annotations"]:
        per_image_cats.setdefault(ann["image_id"], []).append(ann["category_id"])

    filepaths, labels = [], []
    raw_counts = Counter()
    for img_id, file_name in img_id_to_file.items():
        cats = per_image_cats.get(img_id, [])
        if not cats:
            continue
        # majority raw category for this image
        raw_name = cat_id_to_name[Counter(cats).most_common(1)[0][0]]
        raw_counts[raw_name] += 1
        disease = map_category_to_disease(raw_name)
        if disease is None:
            continue
        path = os.path.join(split_dir, file_name)
        if os.path.exists(path):
            filepaths.append(path)
            labels.append(DISEASE_CLASSES.index(disease))

    return filepaths, labels, raw_counts


def summarize(dataset_dir: str):
    """Print what was found — run this first to sanity-check the mapping."""
    splits = find_splits(dataset_dir)
    if not splits:
        print("No COCO splits found. Check the dataset path.")
        return
    for split, path in splits.items():
        fps, labels, raw = load_split(path)
        by_class = Counter(DISEASE_CLASSES[i] for i in labels)
        print(f"[{split}] {len(fps)} usable images  ->  {dict(by_class)}")
        print(f"         raw classes seen: {dict(raw)}")
