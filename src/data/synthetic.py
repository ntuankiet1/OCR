"""Synthetic parcel jobs for demos, CI and pipeline smoke tests.

Renders Thai shipping labels (Code128 barcode, recipient, unit, sender block) onto parcel
bodies shaped and sized after the BRD type/size classes, and writes exactly what the real
systems would hand the pipeline:

  images/<sha256>.jpg          label photos
  review_events.jsonl          review-service export (officer / auto / audit decisions)
  gold_labels.jsonl            two-reviewer ("adjudicated") labels for the frozen sets
  frozen_<split>_keys.txt      group keys for `parcel-ocr freeze`
  <rosters_dir>/<property>.json  resident rosters used for unit/recipient matching

Numbers measured on synthetic data say nothing about real parcels — it only proves that the
pipeline runs end to end. Names, condos and shops are invented.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, features

from src.data.ingest import group_key_for
from src.schemas import LabelField, LabelRecord, LabelSource, ReviewEvent

FIRST = (
    "สมชาย",
    "สมหญิง",
    "วิชัย",
    "สุดารัตน์",
    "ประเสริฐ",
    "อรุณี",
    "ธนพล",
    "กาญจนา",
    "ณัฐวุฒิ",
    "ปิยะนุช",
    "อนุชา",
    "พรทิพย์",
    "ศักดิ์ชัย",
    "วราภรณ์",
    "กิตติ",
    "จันทร์เพ็ญ",
    "ธีรวัฒน์",
    "นภาพร",
    "เอกชัย",
    "รัตนา",
    "สุริยา",
    "มาลี",
    "ชยพล",
    "ศิริพร",
)
LAST = (
    "ใจดี",
    "แสงทอง",
    "ศรีสุข",
    "บุญมา",
    "วงศ์สวัสดิ์",
    "รัตนพันธ์",
    "ทองดี",
    "สุขสวัสดิ์",
    "พรหมมา",
    "เจริญผล",
    "มั่นคง",
    "ศรีวงศ์",
    "แก้วมณี",
    "ชัยมงคล",
    "สมบูรณ์",
    "อินทร์แก้ว",
)
CONDOS = ("สวนสวรรค์ เรสซิเดนซ์", "ริมทาง คอนโดมิเนียม", "ฟ้าใส ทาวเวอร์", "บ้านสบาย เพลส")
SHOPS = (
    "ร้านเสื้อผ้าแฟชั่น",
    "ร้านอุปกรณ์ไอที",
    "ร้านหนังสือดีดี",
    "ร้านของเล่นเด็ก",
    "ร้านเครื่องสำอาง",
    "ร้านอาหารแห้ง",
)
DISTRICTS = (
    "แขวงคลองเตย เขตคลองเตย กรุงเทพฯ 10110",
    "แขวงสามเสนใน เขตพญาไท กรุงเทพฯ 10400",
    "ตำบลบางกระสอ อำเภอเมือง นนทบุรี 11000",
)

# Synthetic-only formats. Flash numbers are deliberately NOT in configs/carriers.yaml, so the
# rule extractor cannot trust them and those jobs route to review — as unknown formats should.
CARRIERS = {
    "thailand_post": ("ไปรษณีย์ไทย  THAILAND POST", 0.8),
    "flash": ("Flash Express  แฟลช เอ็กซ์เพรส", 0.2),
}

# type/size id → (kind, body size relative to the canvas)
SHAPES = {
    1: ("envelope", (0.38, 0.26)),
    2: ("envelope", (0.52, 0.36)),
    3: ("envelope", (0.68, 0.48)),
    4: ("bag", (0.42, 0.36)),
    5: ("bag", (0.58, 0.50)),
    6: ("bag", (0.76, 0.66)),
    7: ("box", (0.36, 0.32)),
    8: ("box", (0.50, 0.44)),
    9: ("box", (0.66, 0.58)),
    10: ("box", (0.86, 0.80)),
}
COLORS = {
    "envelope": ((240, 242, 245), (140, 196, 222)),
    "bag": ((75, 70, 70), (160, 150, 150), (35, 35, 35)),
    "box": ((84, 132, 176), (104, 154, 196)),
}  # BGR
CANVAS = 1024
FONT_GLOBS = (
    "**/Garuda*.[ot]tf",
    "**/Loma*.[ot]tf",
    "**/Sarabun*.ttf",
    "**/NotoSansThai*.ttf",
    "**/Kinnari*.[ot]tf",
    "**/Norasi*.[ot]tf",
    "**/Umpush*.[ot]tf",
)
FONT_DIRS = ("/usr/share/fonts", "/usr/local/share/fonts", "~/.fonts", "~/.local/share/fonts")

Polygon = list[tuple[float, float]]


def s10_tracking(rng: random.Random) -> str:
    """UPU S10 number with a valid check digit, e.g. EF123456785TH."""
    serial = f"{rng.randrange(10**8):08d}"
    c = 11 - sum(int(d) * w for d, w in zip(serial, (8, 6, 4, 2, 3, 5, 9, 7), strict=True)) % 11
    check = {10: 0, 11: 5}.get(c, c)
    return f"{rng.choice(('EF', 'EB', 'RB', 'EQ'))}{serial}{check}TH"


def find_thai_fonts() -> list[Path]:
    if os.environ.get("PARCEL_OCR_THAI_FONT"):
        return [Path(os.environ["PARCEL_OCR_THAI_FONT"])]
    found = {
        Path(p)
        for d in FONT_DIRS
        for g in FONT_GLOBS
        for p in glob.glob(os.path.join(os.path.expanduser(d), g), recursive=True)
        if "Oblique" not in p and "Italic" not in p
    }
    if not found:
        raise FileNotFoundError(
            "no Thai font found: install fonts-thai-tlwg (apt) or set PARCEL_OCR_THAI_FONT"
        )
    return sorted(found)


def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    # Raqm shapes Thai properly (stacked tone marks); BASIC is an acceptable fallback.
    layout = ImageFont.Layout.RAQM if features.check("raqm") else ImageFont.Layout.BASIC
    return ImageFont.truetype(str(path), size, layout_engine=layout)


def _text(
    draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font, pad: int = 3
) -> Polygon:
    draw.text(xy, text, font=font, fill=(0, 0, 0))
    left, top, right, bottom = draw.textbbox(xy, text, font=font)
    left, top, right, bottom = left - pad, top - pad, right + pad, bottom + pad
    return [(left, top), (right, top), (right, bottom), (left, bottom)]


def _barcode(text: str, size: tuple[int, int]) -> Image.Image:
    import barcode
    from barcode.writer import ImageWriter

    img = barcode.get("code128", text, writer=ImageWriter()).render(
        writer_options={"write_text": False, "module_height": 12.0, "quiet_zone": 2.0}
    )
    return img.convert("RGB").resize(size, Image.NEAREST)


# --------------------------------------------------------------------------- rosters
def build_rosters(n_properties: int, units_per_property: int, seed: int) -> dict[str, list[dict]]:
    rng = random.Random(f"roster-{seed}")
    rosters = {}
    for p in range(1, n_properties + 1):
        pid = f"P{p}"
        labels = sorted(
            {
                f"{rng.choice('ABC')}/{rng.randint(2, 32):02d}{rng.randint(1, 20):02d}"
                for _ in range(units_per_property * 2)
            }
        )[:units_per_property]
        units = []
        for i, label in enumerate(labels):
            names = rng.sample([f"{f} {last}" for f in FIRST for last in LAST], rng.randint(1, 3))
            units.append(
                {
                    "unit_id": f"{pid}-{label}",
                    "labels": [label],
                    "residents": [
                        {
                            "recipient_id": f"{pid}-R{i:04d}-{j}",
                            "names": [n],
                            "has_account": rng.random() > 0.05,
                        }
                        for j, n in enumerate(names)
                    ],
                }
            )
        rosters[pid] = units
    return rosters


# --------------------------------------------------------------------------- rendering
@dataclass
class Job:
    job_id: str
    property_id: str
    carrier: str
    tracking: str
    name: str
    unit: str
    condo: str
    district: str
    phone_last4: str
    shop: str
    shop_house: str
    type_size: int
    unit_id: str
    recipient_id: str | None


def render_label(
    rng: random.Random, font_path: Path, job: Job
) -> tuple[Image.Image, dict[str, Polygon]]:
    w, h = 640, 420
    img = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    big, mid, small = _font(font_path, 28), _font(font_path, 24), _font(font_path, 19)
    d.rectangle([2, 2, w - 3, h - 3], outline=(0, 0, 0), width=2)
    d.text((16, 12), CARRIERS[job.carrier][0], font=mid, fill=(0, 0, 0))
    img.paste(_barcode(job.tracking, (300, 66)), (w - 320, 50))

    polys = {"tracking_number": _text(d, (w - 300, 122), job.tracking, small)}
    y = 172
    d.text((16, y), "ผู้รับ:", font=big, fill=(0, 0, 0))
    polys["recipient_name"] = _text(d, (16 + d.textlength("ผู้รับ: ", font=big), y), job.name, big)
    y += 42
    d.text((16, y), "ห้อง", font=mid, fill=(0, 0, 0))
    polys["unit"] = _text(d, (16 + d.textlength("ห้อง ", font=mid), y), job.unit, mid)
    d.text((polys["unit"][1][0] + 12, y), job.condo, font=mid, fill=(0, 0, 0))
    y += 36
    d.text((16, y), job.district, font=small, fill=(0, 0, 0))
    y += 28
    d.text((16, y), f"โทร 08x-xxx-{job.phone_last4}", font=small, fill=(0, 0, 0))
    y += 34
    d.line([12, y, w - 12, y], fill=(0, 0, 0), width=1)
    y += 10
    d.text((16, y), f"ผู้ส่ง: {job.shop}", font=small, fill=(0, 0, 0))
    y += 26  # sender house number looks like a unit on purpose: the extractor must ignore it
    d.text((16, y), f"{job.shop_house} ถนนพระราม 4 กรุงเทพฯ 10500", font=small, fill=(0, 0, 0))
    return img, polys


def compose(
    rng: random.Random, label: Image.Image, polys: dict[str, Polygon], type_size: int
) -> tuple[np.ndarray, dict[str, Polygon]]:
    kind, (rw, rh) = SHAPES[type_size]
    floor = np.array([rng.randint(150, 210)] * 3, dtype=np.float32)
    canvas = np.clip(
        floor + np.random.default_rng(rng.randrange(2**32)).normal(0, 6, (CANVAS, CANVAS, 3)),
        0,
        255,
    ).astype(np.uint8)
    bw, bh = int(rw * CANVAS * rng.uniform(0.92, 1.05)), int(rh * CANVAS * rng.uniform(0.92, 1.05))
    x0, y0 = rng.randint(0, CANVAS - bw), rng.randint(0, CANVAS - bh)
    color = tuple(int(c) for c in rng.choice(COLORS[kind]))
    if kind == "bag":
        r = min(bw, bh) // 6
        cv2.rectangle(canvas, (x0 + r, y0), (x0 + bw - r, y0 + bh), color, -1)
        cv2.rectangle(canvas, (x0, y0 + r), (x0 + bw, y0 + bh - r), color, -1)
        for cx, cy in (
            (x0 + r, y0 + r),
            (x0 + bw - r, y0 + r),
            (x0 + r, y0 + bh - r),
            (x0 + bw - r, y0 + bh - r),
        ):
            cv2.circle(canvas, (cx, cy), r, color, -1)
    else:
        cv2.rectangle(canvas, (x0, y0), (x0 + bw, y0 + bh), color, -1)
        if kind == "box":  # packing tape
            tw = max(12, bw // 10)
            cv2.rectangle(
                canvas,
                (x0 + bw // 2 - tw // 2, y0),
                (x0 + bw // 2 + tw // 2, y0 + bh),
                (150, 190, 205),
                -1,
            )
        else:  # envelope flap
            cv2.line(canvas, (x0, y0), (x0 + bw // 2, y0 + bh // 3), (120, 120, 120), 2)
            cv2.line(canvas, (x0 + bw, y0), (x0 + bw // 2, y0 + bh // 3), (120, 120, 120), 2)

    scale = min(0.9 * bw / label.width, 0.9 * bh / label.height, 1.0) * rng.uniform(0.85, 1.0)
    lw, lh = int(label.width * scale), int(label.height * scale)
    lx, ly = x0 + rng.randint(0, bw - lw), y0 + rng.randint(0, bh - lh)
    small = cv2.cvtColor(np.asarray(label.resize((lw, lh), Image.LANCZOS)), cv2.COLOR_RGB2BGR)
    canvas[ly : ly + lh, lx : lx + lw] = small
    polys = {k: [(lx + x * scale, ly + y * scale) for x, y in p] for k, p in polys.items()}

    angle = rng.uniform(-4, 4)
    m = cv2.getRotationMatrix2D((CANVAS / 2, CANVAS / 2), angle, 1.0)
    canvas = cv2.warpAffine(canvas, m, (CANVAS, CANVAS), borderMode=cv2.BORDER_REPLICATE)
    polys = {
        k: [tuple((m @ np.array([x, y, 1.0])).tolist()) for x, y in p] for k, p in polys.items()
    }
    if rng.random() < 0.2:
        canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
    canvas = cv2.convertScaleAbs(canvas, alpha=rng.uniform(0.85, 1.1), beta=rng.uniform(-15, 15))
    return canvas, polys


# --------------------------------------------------------------------------- jobs + exports
def _corrupt(rng: random.Random, text: str) -> str:
    """Simulate a model mistake: one wrong character."""
    if not text:
        return text
    i = rng.randrange(len(text))
    pool = "0123456789" if text[i].isdigit() else "กขคงจฉชซดตถทนบปผพฟมยรลวสหอ"
    return text[:i] + rng.choice(pool.replace(text[i], "") or "x") + text[i + 1 :]


def generate(
    out_dir: Path,
    rosters_dir: Path,
    n_jobs: int,
    seed: int = 7,
    batch: int = 1,
    regression_share: float = 0.15,
    acceptance_share: float = 0.10,
    n_properties: int = 2,
    units_per_property: int = 120,
) -> dict:
    rng = random.Random(f"jobs-{seed}-{batch}")
    fonts = find_thai_fonts()
    rosters = build_rosters(n_properties, units_per_property, seed)
    rosters_dir.mkdir(parents=True, exist_ok=True)
    for pid, units in rosters.items():
        (rosters_dir / f"{pid}.json").write_text(
            json.dumps({"units": units}, ensure_ascii=False, indent=1)
        )

    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    start = datetime(2026, 10, 1, 8, tzinfo=UTC) + timedelta(days=30 * (batch - 1))
    events, gold, frozen = [], [], {"regression": [], "acceptance": []}
    for i in range(n_jobs):
        pid = rng.choice(sorted(rosters))
        unit = rng.choice(rosters[pid])
        known = rng.random() > 0.05  # 5%: recipient not in the roster → must go to review
        if known:
            res = rng.choice(unit["residents"])
            name, recipient_id = res["names"][0], res["recipient_id"]
        else:
            name, recipient_id = f"{rng.choice(FIRST)} {rng.choice(LAST)}", None
        carrier = rng.choices(list(CARRIERS), weights=[w for _, w in CARRIERS.values()])[0]
        tracking = (
            s10_tracking(rng)
            if carrier == "thailand_post"
            else f"TH{rng.randrange(10**10):010d}{rng.choice('ABCDEFGHJK')}"
        )
        job = Job(
            job_id=f"demo-b{batch}-{i:05d}",
            property_id=pid,
            carrier=carrier,
            tracking=tracking,
            name=name,
            unit=unit["labels"][0],
            condo=rng.choice(CONDOS),
            district=rng.choice(DISTRICTS),
            phone_last4=f"{rng.randrange(10**4):04d}",
            shop=rng.choice(SHOPS),
            shop_house=f"{rng.randint(1, 99)}/{rng.randint(1, 9)}",
            type_size=rng.randint(1, 10),
            unit_id=unit["unit_id"],
            recipient_id=recipient_id,
        )

        label, polys = render_label(rng, rng.choice(fonts), job)
        image, polys = compose(rng, label, polys, job.type_size)
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(75, 95)])
        data = buf.tobytes()
        sha = hashlib.sha256(data).hexdigest()
        path = (out_dir / "images" / f"{sha}.jpg").resolve()
        path.write_bytes(data)

        truth = {
            LabelField.TRACKING_NUMBER: (job.tracking, polys["tracking_number"]),
            LabelField.RECIPIENT_NAME: (job.name, polys["recipient_name"]),
            LabelField.UNIT: (job.unit, polys["unit"]),
            LabelField.CARRIER: (job.carrier, None),
            LabelField.ADDRESS: (f"{job.unit} {job.condo} {job.district}", None),
            LabelField.TYPE_SIZE: (str(job.type_size), None),
            LabelField.UNIT_ID: (job.unit_id, None),
            LabelField.RECIPIENT_ID: (job.recipient_id, None),
        }
        final = {f: {"value": v, "polygon": p} for f, (v, p) in truth.items()}
        predicted = {f: dict(fv) for f, fv in final.items()}
        for f, p_err in (
            (LabelField.RECIPIENT_NAME, 0.15),
            (LabelField.UNIT, 0.08),
            (LabelField.TRACKING_NUMBER, 0.03),
        ):
            if rng.random() < p_err:
                predicted[f]["value"] = _corrupt(rng, predicted[f]["value"])
        if rng.random() < 0.25:
            predicted[LabelField.TYPE_SIZE]["value"] = str(rng.randint(1, 10))

        roll = rng.random()
        decided_by = "auto" if roll < 0.10 else "audit" if roll < 0.13 else "officer"
        ts = start + timedelta(minutes=3 * i)
        event = ReviewEvent.model_validate(
            {
                "job_id": job.job_id,
                "image_sha256": sha,
                "image_uri": str(path),
                "property_id": pid,
                "capture_mode": "auto_camera" if rng.random() < 0.3 else "officer",
                "captured_at": ts,
                "decided_at": ts + timedelta(seconds=40),
                "decided_by": decided_by,
                "reviewer_id": None if decided_by == "auto" else f"officer-{rng.randint(1, 4)}",
                "bundle_id": "synthetic",
                "predicted": predicted,
                "final": final,
            }
        )
        events.append(event)

        split = None
        if batch == 1:
            u = rng.random()
            split = (
                "regression"
                if u < regression_share
                else "acceptance"
                if u < regression_share + acceptance_share
                else None
            )
        if split:
            frozen[split].append(group_key_for(event))
            for f, (v, p) in truth.items():
                gold.append(
                    LabelRecord(
                        label_id=LabelRecord.make_id(job.job_id, f, "adjudicated", v),
                        image_sha256=sha,
                        field=f,
                        value=v,
                        polygon=p,
                        source=LabelSource.ADJUDICATED,
                        author_id="reviewer-pair-1",
                        created_at=ts + timedelta(days=1),
                    )
                )

    (out_dir / "review_events.jsonl").write_text(
        "".join(e.model_dump_json() + "\n" for e in events), encoding="utf-8"
    )
    (out_dir / "gold_labels.jsonl").write_text(
        "".join(g.model_dump_json() + "\n" for g in gold), encoding="utf-8"
    )
    for split, keys in frozen.items():
        (out_dir / f"frozen_{split}_keys.txt").write_text("".join(k + "\n" for k in keys))
    summary = {
        "batch": batch,
        "jobs": n_jobs,
        "gold_labels": len(gold),
        "frozen_groups": {k: len(v) for k, v in frozen.items()},
        "decided_by": {
            k: sum(e.decided_by == k for e in events) for k in ("officer", "auto", "audit")
        },
        "fonts": [f.name for f in fonts],
        "raqm": features.check("raqm"),
        "out_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary
