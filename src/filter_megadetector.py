from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from PytorchWildlife.models import detection as pw_detection

ANIMAL_CLASS_ID = 0
DEFAULT_EXTS = (".jpeg", ".jpg", ".png")


def iter_image_files(root: Path, exts: tuple[str, ...]):
    for folder in root.iterdir():
        if folder.is_dir() and folder.name.endswith("_images"):
            for f in folder.iterdir():
                if f.is_file() and f.suffix.lower() in exts:
                    yield f


def has_animal(detector, image_path: Path, threshold: float) -> bool:
    result = detector.single_image_detection(str(image_path))
    dets = result.get("detections")
    if dets is None:
        return False
    for conf, cls in zip(dets.confidence, dets.class_id):
        if cls == ANIMAL_CLASS_ID and conf >= threshold:
            return True
    return False


def relocate_non_animal(
    input_root: Path, threshold: float, exts: tuple[str, ...], limit: int | None
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = pw_detection.MegaDetectorV5(device=device, pretrained=True, version="a")
    scanned = animal_present = moved = 0
    images = list(iter_image_files(input_root, exts))
    if limit is not None:
        images = images[:limit]
    output_root = input_root / "not_detected"
    for img in images:
        scanned += 1
        if has_animal(detector, img, threshold):
            animal_present += 1
            continue
        target_dir = output_root / img.parent.name
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / img.name
        shutil.move(str(img), str(target_file))
        print(f"Moved {img} -> {target_file}")
        moved += 1
    print(
        f"Done. Scanned={scanned} animal_present={animal_present} moved_not_detected={moved}"
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Relocate images without animal detections using MegaDetectorV5-a."
    )
    p.add_argument("--input-root", type=Path, default=Path("iNaturalist"))
    p.add_argument("--threshold", type=float, default=0.2)
    p.add_argument("--exts", nargs="*", default=list(DEFAULT_EXTS))
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    exts = tuple(e if e.startswith(".") else f".{e}" for e in args.exts)
    relocate_non_animal(args.input_root, args.threshold, exts, args.limit)


if __name__ == "__main__":
    main()
