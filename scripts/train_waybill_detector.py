"""Train a nano YOLO that finds the waybill (one class) in a parcel photo.

Three steps, each its own subcommand so you can rerun just one:

  python scripts/train_waybill_detector.py download --out waybill_det      # Roboflow -> YOLO folders
  python scripts/train_waybill_detector.py eda      --out waybill_det      # merge + stats + plots
  python scripts/train_waybill_detector.py train    --out waybill_det      # fine-tune yolo26n

Needs: pip install roboflow "ultralytics>=8.4" pillow matplotlib pyyaml
and a free Roboflow API key in ROBOFLOW_API_KEY (Settings -> API Keys).

Sources were picked by looking at their images (see SOURCES). Per source only the whole-label class
is kept (barcode / address / QR sub-boxes are dropped) and everything is merged into one class
`waybill`. Roboflow's own train/valid/test splits are kept, because its augmented copies exist only in
train; re-splitting at random would leak copies of one photo into the test set. Images are de-duplicated
by content hash. Sources marked "test" (few images, closest to Thai couriers) are only used for testing.
NOTE: Ultralytics YOLO is AGPL-3.0; for a closed commercial service use RF-DETR (Apache-2.0)
or buy the Ultralytics enterprise licence. The data and the `eda` step are reusable either way.
"""

import argparse
import hashlib
import json
import os
import random
import shutil
from pathlib import Path

# (workspace/project/version, export format, whole-label classes to keep (None = all), role)
# Checked by eye: real photos of parcels with a label box. Left out: johnson-tseng-o0zxe/shipping-label
# (synthetic flat PDF pages), test-run-s13my/shipping-labels-or-not (food/produce stickers),
# packageandbarcode/package-and-label-detection-agjvl (its `label` class is merged into `package`).
SOURCES = [
    ("soorya-s-workspace/shipping-labels-ee58q/1", "yolov8", None, "train"),  # handheld parcels
    ("hector-yanez/shipping-label-detection-8a6g7/1", "yolov8", None, "train"),  # close-ups, US carriers
    ("tim-4ijf0/label-detection-civy2/1", "yolov5", {"shipping-label"}, "train"),
    # Philippine Flash / SPX / J&T parcels; only 8 images carry a whole-label `Waybill` box
    ("nextgenartisans/private-x9tv4/7", "yolov8", {"Waybill"}, "test"),
]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def source_dir(out: Path, spec: str) -> Path:
    return out / "raw" / spec.split("/")[1]


def download(args):
    from roboflow import Roboflow

    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise SystemExit("set ROBOFLOW_API_KEY (Roboflow -> Settings -> API Keys)")
    rf = Roboflow(api_key=key)
    for spec, fmt, _, _ in SOURCES:
        ws, proj, ver = spec.split("/")
        dest = source_dir(args.out, spec)
        if dest.exists() and any(dest.iterdir()):
            print(f"skip {spec} (already in {dest})")
            continue
        print(f"downloading {spec} -> {dest}")
        # "yolov8" is the plain YOLO txt layout (boxes or polygons), usable by any detector
        rf.workspace(ws).project(proj).version(int(ver)).download(fmt, location=str(dest))


def to_box(parts: list[str]):
    """One YOLO label line -> (cx, cy, w, h) normalised, from a box or a polygon."""
    v = [float(x) for x in parts[1:]]
    if len(v) == 4:
        return tuple(v)
    xs, ys = v[0::2], v[1::2]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return ((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)


def collect(out: Path):
    """(source, split, image path, [boxes]) with only the kept class, de-duplicated by hash."""
    import yaml

    seen, items, dupes = set(), [], 0
    for spec, _, keep, role in SOURCES:
        src = source_dir(out, spec)
        if not src.exists():
            print(f"missing {src}, run `download`")
            continue
        names = yaml.safe_load((src / "data.yaml").read_text())["names"]
        names = list(names.values()) if isinstance(names, dict) else names
        for img in sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT):
            lab = img.parent.parent / "labels" / (img.stem + ".txt")
            if not lab.exists():
                continue
            digest = hashlib.md5(img.read_bytes()).hexdigest()
            if digest in seen:
                dupes += 1
                continue
            seen.add(digest)
            boxes = []
            for line in lab.read_text().splitlines():
                parts = line.split()
                if len(parts) >= 5 and (keep is None or names[int(parts[0])] in keep):
                    boxes.append(to_box(parts))
            split = "test" if role == "test" else {"valid": "val"}.get(img.parent.parent.name,
                                                                         img.parent.parent.name)
            items.append((src.name, split, img, boxes))
    return items, dupes


def eda(args):
    from PIL import Image

    items, dupes = collect(args.out)
    if not items:
        raise SystemExit("no labelled images found; run `download` first")

    per_src, kept = {}, []
    for src, split, img, boxes in items:
        w, h = Image.open(img).size
        # drop images with no kept box (nothing to learn) and tiny/degenerate boxes
        boxes = [b for b in boxes if b[2] * w >= 16 and b[3] * h >= 16]
        if boxes:
            per_src[src] = per_src.get(src, 0) + 1
            kept.append((src, split, img, boxes, w, h))

    n = len(kept)
    split = {k: [r for r in kept if r[1] == k] for k in ("train", "val", "test")}
    data = args.out / "data"
    if data.exists():
        shutil.rmtree(data)
    for name, rows_ in split.items():
        (data / "images" / name).mkdir(parents=True)
        (data / "labels" / name).mkdir(parents=True)
        for i, (src, _, img, boxes, _, _) in enumerate(rows_):
            stem = f"{src}_{i:05d}"
            shutil.copy(img, data / "images" / name / (stem + img.suffix.lower()))
            (data / "labels" / name / (stem + ".txt")).write_text(
                "".join(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n" for cx, cy, bw, bh in boxes))
    (args.out / "data.yaml").write_text(
        f"path: {(args.out / 'data').resolve()}\ntrain: images/train\nval: images/val\n"
        "test: images/test\nnames:\n  0: waybill\n")

    areas = [bw * bh for _, _, _, bs, _, _ in kept for _, _, bw, bh in bs]
    aspect = [(bw * w) / (bh * h) for _, _, _, bs, w, h in kept for _, _, bw, bh in bs]
    per_img = [len(bs) for _, _, _, bs, _, _ in kept]
    sizes = [max(w, h) for _, _, _, _, w, h in kept]

    def pct(v, q):
        v = sorted(v)
        return round(v[int(q * (len(v) - 1))], 4)

    report = {
        "images_found": len(items), "duplicates_dropped": dupes,
        "images_kept": n, "images_per_source": per_src,
        "split": {k: len(v) for k, v in split.items()},
        "boxes": len(areas),
        "boxes_per_image": {"mean": round(sum(per_img) / n, 2), "max": max(per_img),
                            "multi_box_images": sum(c > 1 for c in per_img)},
        "box_area_fraction": {q: pct(areas, q) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
        "box_aspect_w_over_h": {q: pct(aspect, q) for q in (0.05, 0.5, 0.95)},
        "image_long_side": {q: int(pct(sizes, q)) for q in (0.05, 0.5, 0.95)},
        "small_boxes_lt_2pct_of_image": sum(a < 0.02 for a in areas),
    }
    (args.out / "eda_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 4, figsize=(18, 3.6))
        ax[0].bar(per_src.keys(), per_src.values()); ax[0].set_title("images per source")
        ax[0].tick_params(axis="x", labelrotation=20, labelsize=7)
        ax[1].hist(areas, bins=30); ax[1].set_title("box area / image area")
        ax[2].hist([a for a in aspect if a < 6], bins=30); ax[2].set_title("box w/h")
        ax[3].hist(sizes, bins=30); ax[3].set_title("image long side (px)")
        fig.tight_layout(); fig.savefig(args.out / "eda.png", dpi=110)

        from PIL import ImageDraw

        tiles = []
        for _, _, img, boxes, w, h in kept[:12]:
            im = Image.open(img).convert("RGB"); d = ImageDraw.Draw(im)
            for cx, cy, bw, bh in boxes:
                d.rectangle([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w,
                             (cy + bh / 2) * h], outline=(255, 0, 0), width=max(3, w // 150))
            tiles.append(im.resize((320, 320)))
        sheet = Image.new("RGB", (320 * 4, 320 * 3))
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % 4) * 320, (i // 4) * 320))
        sheet.save(args.out / "samples.jpg", quality=90)
        print(f"wrote {args.out / 'eda.png'} and {args.out / 'samples.jpg'}")
    except ImportError:
        print("matplotlib missing: skipped plots")


def train(args):
    from ultralytics import YOLO

    model = YOLO(args.weights)
    model.train(data=str(args.out / "data.yaml"), imgsz=args.imgsz, epochs=args.epochs,
                batch=args.batch, patience=15, project=str(args.out / "runs"), name="waybill",
                exist_ok=True, seed=0)
    best = args.out / "runs" / "waybill" / "weights" / "best.pt"
    print("test split:", YOLO(str(best)).val(data=str(args.out / "data.yaml"), split="test").box.map50,
          "mAP50")
    print(f"weights: {best}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("download", download), ("eda", eda), ("train", train)):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, default=Path("waybill_det"))
        p.set_defaults(fn=fn)
        if name == "train":
            p.add_argument("--weights", default="yolo26n.pt")
            p.add_argument("--imgsz", type=int, default=640)
            p.add_argument("--epochs", type=int, default=100)
            p.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
