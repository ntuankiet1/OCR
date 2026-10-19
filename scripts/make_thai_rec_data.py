"""Generate a synthetic Thai+English text-line dataset that looks like parcel labels,
and evaluate whether a dataset fits the parcel-label requirements.

What is generated (PaddleOCR text-recognition format):
  <out>/images/0000001.jpg ...
  <out>/train.txt  <out>/val.txt  <out>/test.txt        lines: "images/0000001.jpg<TAB>text"
  <out>/report.json  <out>/samples.jpg                   dataset evaluation + sample sheet

Content follows parcel labels, not ThaiOCRBench (signage/handwriting would be off-purpose):
Thai and romanized names, unit numbers, address lines, phone numbers, tracking numbers,
carrier names and label words, plus general Thai words for vocabulary coverage.
Rendering follows label conditions: dark print on white/thermal paper, mild blur, fading,
tilt and JPEG noise. Fonts are SIL OFL Thai fonts, looped and loopless (the current model
reads loopless Thai as Latin). test.txt uses fonts never seen in train/val.

Setup:  pip install pythainlp pillow fonttools numpy
        (Linux: Pillow needs libraqm/libfribidi for correct Thai shaping:
         apt-get install -y libraqm0 libfribidi0)
Run:    python scripts/make_thai_rec_data.py --out data/thai_rec_synth --n 100000
        python scripts/make_thai_rec_data.py --evaluate other_dataset/train.txt   # fit check only
"""

import argparse
import json
import random
import re
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, features

FONT_URL = "https://raw.githubusercontent.com/google/fonts/main/ofl/"
FONTS = {  # google/fonts path -> style. All SIL Open Font License (commercial use allowed).
    "kanit/Kanit-Regular.ttf": "loopless",
    "kanit/Kanit-Bold.ttf": "loopless",
    "prompt/Prompt-Regular.ttf": "loopless",
    "prompt/Prompt-Bold.ttf": "loopless",
    "mitr/Mitr-Regular.ttf": "loopless",
    "baijamjuree/BaiJamjuree-Regular.ttf": "loopless",
    "chakrapetch/ChakraPetch-Regular.ttf": "loopless",
    "athiti/Athiti-Regular.ttf": "loopless",
    "notosansthai/NotoSansThai%5Bwdth,wght%5D.ttf": "loopless",
    "ibmplexsansthai/IBMPlexSansThai-Regular.ttf": "loopless",
    "ibmplexsansthai/IBMPlexSansThai-Bold.ttf": "loopless",
    "anuphan/Anuphan%5Bwght%5D.ttf": "loopless",
    "k2d/K2D-Regular.ttf": "loopless",
    "sarabun/Sarabun-Regular.ttf": "looped",
    "sarabun/Sarabun-Bold.ttf": "looped",
    "notosansthailooped/NotoSansThaiLooped%5Bwdth,wght%5D.ttf": "looped",
    "ibmplexsansthailooped/IBMPlexSansThaiLooped-Regular.ttf": "looped",
    "krub/Krub-Regular.ttf": "looped",
    "pridi/Pridi-Regular.ttf": "looped",
    "taviraj/Taviraj-Regular.ttf": "looped",
    "trirong/Trirong-Regular.ttf": "looped",
    "notoserifthai/NotoSerifThai%5Bwdth,wght%5D.ttf": "looped",
    "niramit/Niramit-Regular.ttf": "looped",
}
TEST_FONTS = {"k2d/K2D-Regular.ttf", "niramit/Niramit-Regular.ttf"}  # one loopless, one looped
STYLE_WEIGHTS = {"loopless": 0.45, "looped": 0.35, "tlwg": 0.20}  # share of lines per font style
# Thai system fonts (fonts-thai-tlwg) are added when installed; handwriting styles are skipped.
TLWG_DIRS = ["/usr/share/fonts/opentype/tlwg", "/usr/share/fonts/truetype/tlwg"]
# Skipped: handwriting styles, slanted variants, and the monospace family — TlwgMono places
# Thai marks wrongly (image would not match its label) and typewriter fonts are not label print.
TLWG_SKIP = (
    "Purisa",
    "Sawasdee",
    "Oblique",
    "Italic",
    "TlwgMono",
    "TlwgTypo",
    "TlwgTypist",
    "TlwgTypewriter",
)
DICT_URL = "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/v3.7.0/ppocr/utils/dict/ppocrv5_th_dict.txt"
MAX_LEN = 25  # PaddleOCR rec configs: max_text_length 25 (longer labels are dropped in training)

# Share of each content type, chosen to mirror what is printed on parcel labels.
MIX = {
    "thai_name": 0.16,
    "latin_name": 0.08,
    "unit": 0.08,
    "address": 0.16,
    "phone": 0.07,
    "tracking": 0.09,
    "label_words": 0.10,
    "thai_words": 0.18,
    "english_words": 0.04,
    "mixed": 0.04,
}

HONORIFICS = ["คุณ", "นาย", "นาง", "นางสาว", "น.ส.", "", "", ""]
EN_HONORIFICS = ["Mr.", "Ms.", "Mrs.", "Khun", "", "", ""]
CARRIERS = [
    "Flash Express",
    "Kerry Express",
    "KEX Express",
    "J&T Express",
    "Thailand Post",
    "SPX Express",
    "Shopee Xpress",
    "Lazada Express",
    "Ninja Van",
    "DHL eCommerce",
    "Best Express",
    "ไปรษณีย์ไทย",
    "แฟลช เอ็กซ์เพรส",
    "เคอรี่ เอ็กซ์เพรส",
    "EMS",
    "ลาซาด้า",
]
LABEL_WORDS = [
    "ผู้รับ",
    "ผู้ส่ง",
    "ที่อยู่",
    "โทร",
    "เบอร์โทร",
    "ชื่อผู้รับ",
    "ชื่อผู้ส่ง",
    "เก็บเงินปลายทาง",
    "COD",
    "น้ำหนัก",
    "ระวังแตก",
    "ห้ามโยน",
    "ส่งด่วน",
    "พัสดุ",
    "รหัสไปรษณีย์",
    "หมายเลขพัสดุ",
    "Tracking No.",
    "Order ID",
    "Weight",
    "Ship To",
    "From",
    "To",
    "Sender",
    "Receiver",
    "Recipient",
    "Tel",
    "Phone",
    "Qty",
    "ชิ้น",
    "บาท",
    "กก.",
    "kg",
    "Return to",
]
SOI_ROAD = [
    "ซอย",
    "ซ.",
    "ถนน",
    "ถ.",
    "หมู่",
    "ม.",
    "ตำบล",
    "ต.",
    "อำเภอ",
    "อ.",
    "แขวง",
    "เขต",
    "จังหวัด",
    "จ.",
]
BKK_KHET = [
    "คลองเตย",
    "วัฒนา",
    "บางรัก",
    "ปทุมวัน",
    "จตุจักร",
    "ห้วยขวาง",
    "บางนา",
    "พระโขนง",
    "ดินแดง",
    "สาทร",
    "บางกะปิ",
    "ลาดพร้าว",
    "บางซื่อ",
    "ราชเทวี",
    "พญาไท",
    "บางเขน",
    "สวนหลวง",
    "ประเวศ",
]
ROADS = [
    "สุขุมวิท",
    "พระราม 4",
    "พระราม 9",
    "รัชดาภิเษก",
    "ลาดพร้าว",
    "พหลโยธิน",
    "เพชรบุรี",
    "สีลม",
    "งามวงศ์วาน",
    "แจ้งวัฒนะ",
    "รามคำแหง",
    "บางนา-ตราด",
    "ศรีนครินทร์",
    "วิภาวดีรังสิต",
]
EN_PLACES = [
    "Bangkok",
    "Nonthaburi",
    "Pathum Thani",
    "Samut Prakan",
    "Chiang Mai",
    "Chon Buri",
    "Phuket",
    "Khon Kaen",
    "Sukhumvit Rd.",
    "Rama 9 Rd.",
    "Ratchadaphisek Rd.",
    "Khlong Toei",
    "Watthana",
    "Bang Na",
    "Huai Khwang",
    "Chatuchak",
    "Thailand",
    "Soi",
    "Moo",
    "Condo",
]
CONDO_WORDS = [
    "คอนโด",
    "อาคาร",
    "ตึก",
    "ชั้น",
    "ห้อง",
    "เลขที่",
    "Tower",
    "Building",
    "Floor",
    "Room",
    "Unit",
]
EN_WORDS = [
    "fragile",
    "express",
    "standard",
    "parcel",
    "package",
    "delivery",
    "return",
    "office",
    "online",
    "shop",
    "store",
    "official",
    "cosmetics",
    "fashion",
    "electronics",
    "books",
    "home",
    "kitchen",
    "baby",
    "sport",
    "health",
    "beauty",
    "mobile",
    "accessories",
    "gift",
]

THAI = re.compile(r"[฀-๿]")
LATIN = re.compile(r"[A-Za-z]")
DIGIT = re.compile(r"[0-9๐-๙]")


# --------------------------------------------------------------------------- text content
class Corpus:
    def __init__(self, rng: random.Random):
        from pythainlp.corpus import (
            provinces,
            thai_family_names,
            thai_female_names,
            thai_male_names,
            thai_words,
        )
        from pythainlp.transliterate import romanize

        self.rng = rng
        self.first = sorted(thai_female_names() | thai_male_names())
        self.family = sorted(thai_family_names())
        self.words = sorted(w for w in thai_words() if 1 < len(w) <= 12)
        self.provinces = sorted(provinces())
        self.romanize = romanize

    def pick(self, seq):
        return self.rng.choice(seq)

    def digits(self, n: int) -> str:
        return "".join(self.rng.choice("0123456789") for _ in range(n))

    def thai_name(self) -> str:
        return f"{self.pick(HONORIFICS)}{self.pick(self.first)} {self.pick(self.family)}".strip()

    def latin_name(self) -> str:
        first = self.romanize(self.pick(self.first), engine="royin").capitalize()
        last = self.romanize(self.pick(self.family), engine="royin")
        last = last.upper() if self.rng.random() < 0.3 else last.capitalize()
        return f"{self.pick(EN_HONORIFICS)} {first} {last}".strip()

    def unit(self) -> str:
        r, ri, ch = self.rng.random(), self.rng.randint, self.rng.choice
        if r < 0.3:
            return f"{ch('ABCDE')}/{ri(1, 45):02d}{ri(1, 30):02d}"
        if r < 0.55:
            return f"{ri(1, 999)}/{ri(1, 999)}"
        if r < 0.8:
            return f"{self.pick(CONDO_WORDS)} {ch('ABCD')}{ri(1, 45)}{self.digits(2)}"
        return f"ห้อง {ri(1, 45)}{self.digits(2)} ชั้น {ri(1, 45)}"

    def address(self) -> str:
        r, ri, ch = self.rng.random(), self.rng.randint, self.rng.choice
        if r < 0.3:
            return f"{ri(1, 999)}/{ri(1, 99)} {self.pick(SOI_ROAD)}{self.pick(ROADS)}"
        if r < 0.55:
            return f"แขวง{self.pick(BKK_KHET)} เขต{self.pick(BKK_KHET)}"
        if r < 0.75:
            return (
                f"{self.pick(self.provinces)} {ch(['10', '11', '12', '20', '50'])}{self.digits(3)}"
            )
        if r < 0.9:
            return f"{self.pick(SOI_ROAD)}{self.pick(self.words)} {ri(1, 120)}"
        return f"{ri(1, 999)} {self.pick(EN_PLACES)} {self.digits(5)}"

    def phone(self) -> str:
        p = f"0{self.rng.choice('689')}{self.digits(8)}"
        r = self.rng.random()
        if r < 0.35:
            return f"{p[:3]}-{p[3:6]}-{p[6:]}"
        if r < 0.55:
            return p
        if r < 0.7:
            return f"+66 {p[1:3]} {p[3:6]} {p[6:]}"
        if r < 0.85:
            return f"{self.pick(['โทร', 'Tel', 'Tel.', 'เบอร์'])} {p[:3]}-xxx-{p[6:]}"
        return f"{p[:3]}{'*' * 3}{p[6:]}"

    def tracking(self) -> str:
        r = self.rng.random()
        if r < 0.25:
            return f"{self.rng.choice(['EF', 'EB', 'RB', 'EQ', 'EI', 'RA'])}{self.digits(9)}TH"
        if r < 0.5:
            return f"TH{self.digits(10)}{self.rng.choice('ABCDEFGHJK')}"
        if r < 0.65:
            return f"{self.rng.choice(['KER', 'KEX', 'SPX', 'LEX', 'NJV'])}{self.digits(10)}"
        if r < 0.8:
            return self.digits(self.rng.choice([10, 12, 13, 15]))
        prefix, suffix = self.rng.choice(["TH", "SPXTH", "LXAD"]), self.rng.choice(["", "A", "B"])
        return f"{prefix}{self.digits(8)}{suffix}"

    def label_words(self) -> str:
        r, ri, ch = self.rng.random(), self.rng.randint, self.rng.choice
        if r < 0.35:
            return self.pick(CARRIERS)
        if r < 0.65:
            return f"{self.pick(LABEL_WORDS)}{ch([':', ' ', ' : '])}"
        if r < 0.8:
            return f"{ri(1, 30)}/{ri(1, 12)}/{ch(['2568', '2569', '2025', '2026'])}"
        if r < 0.9:
            return f"{ch(['COD', 'เก็บเงินปลายทาง'])} {ri(49, 4999)} บาท"
        return f"{ch(['น้ำหนัก', 'Weight'])} {ri(1, 30)}.{ri(0, 9)} kg"

    def thai_words(self) -> str:
        return "".join(self.pick(self.words) for _ in range(self.rng.randint(1, 3)))

    def english_words(self) -> str:
        words = [self.pick(EN_WORDS) for _ in range(self.rng.randint(1, 3))]
        text = " ".join(words)
        return (
            text.title()
            if self.rng.random() < 0.5
            else (text.upper() if self.rng.random() < 0.5 else text)
        )

    def mixed(self) -> str:
        return self.rng.choice(
            [
                f"ห้อง {self.unit()}",
                f"ส่งด่วน {self.pick(['EMS', 'Express', 'J&T'])}",
                f"{self.pick(self.provinces)} {self.pick(EN_PLACES)}",
                f"{self.thai_name()} {self.digits(3)}",
                f"Shop {self.pick(self.words)}",
                f"{self.pick(LABEL_WORDS)} {self.tracking()}",
            ]
        )

    def sample(self, kind: str) -> str:
        for _ in range(20):
            text = getattr(self, kind)().strip()
            if 0 < len(text) <= MAX_LEN:
                return text
        return getattr(self, kind)().strip()[:MAX_LEN]


# --------------------------------------------------------------------------- fonts
def download_fonts(cache: Path) -> dict[Path, str]:
    cache.mkdir(parents=True, exist_ok=True)
    fonts = {}
    for rel, style in FONTS.items():
        dst = cache / Path(rel).name.replace("%5B", "[").replace("%5D", "]")
        if not dst.exists():
            urllib.request.urlretrieve(FONT_URL + rel, dst)
        fonts[dst] = style
    for d in TLWG_DIRS:
        for p in sorted(Path(d).glob("*.[ot]tf")) if Path(d).exists() else []:
            if not any(s in p.name for s in TLWG_SKIP):
                fonts[p] = "tlwg"
    return fonts


def font_chars(path: Path) -> set[str]:
    from fontTools.ttLib import TTFont

    return {chr(c) for c in TTFont(str(path), lazy=True).getBestCmap()}


# --------------------------------------------------------------------------- rendering
def render(text: str, font_path: Path, rng: random.Random) -> Image.Image:
    size = rng.randint(26, 46)
    font = ImageFont.truetype(str(font_path), size, layout_engine=ImageFont.Layout.RAQM)
    left, top, right, bottom = font.getbbox(text)
    pad_x, pad_y = rng.randint(3, 18), rng.randint(3, 12)
    w, h = right - left + 2 * pad_x, bottom - top + 2 * pad_y
    # Labels are white/grey paper: one grey level plus a slight camera white-balance tint
    # (at most ±6 per channel), never pastel colours.
    tint = [rng.randint(-6, 6) for _ in range(3)]

    def paper(level: int) -> tuple[int, int, int]:
        return tuple(max(0, min(255, level + t)) for t in tint)

    r = rng.random()
    if r < 0.80:  # thermal / white label: dark text on white or off-white
        bg, fg = paper(rng.randint(222, 252)), (rng.randint(0, 70),) * 3
    elif r < 0.93:  # grey paper, faded print
        bg, fg = paper(rng.randint(170, 222)), (rng.randint(20, 110),) * 3
    else:  # inverted header bars (e.g. COD, carrier name)
        bg, fg = paper(rng.randint(0, 50)), (rng.randint(215, 255),) * 3
    img = Image.new("RGB", (w, h), bg)
    ImageDraw.Draw(img).text((pad_x - left, pad_y - top), text, font=font, fill=fg)

    if rng.random() < 0.5:
        img = img.rotate(rng.uniform(-2.5, 2.5), resample=Image.BICUBIC, expand=True, fillcolor=bg)
    if rng.random() < 0.35:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))
    if rng.random() < 0.25:  # low-resolution camera: downscale then upscale
        f = rng.uniform(0.4, 0.8)
        small = img.resize(
            (max(1, int(img.width * f)), max(1, int(img.height * f))), Image.BILINEAR
        )
        img = small.resize(img.size, Image.BILINEAR)
    if rng.random() < 0.3:
        arr = np.asarray(img).astype(np.float32)
        arr += np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(3, 12), arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img


# --------------------------------------------------------------------------- evaluation
def evaluate(entries: list[dict], dictionary: set[str] | None) -> dict:
    """Requirement-fit report for any list of {"text", optional "kind", "style"}."""
    chars = Counter(ch for e in entries for ch in e["text"])
    total = max(sum(chars.values()), 1)
    thai = sum(n for ch, n in chars.items() if THAI.match(ch) and not DIGIT.match(ch))
    latin = sum(n for ch, n in chars.items() if LATIN.match(ch))
    digit = sum(n for ch, n in chars.items() if DIGIT.match(ch))
    lengths = sorted(len(e["text"]) for e in entries) or [0]
    oov = {
        ch: n
        for ch, n in chars.items()
        if dictionary is not None and ch not in dictionary and ch != " "
    }
    styles = Counter(e.get("style", "unknown") for e in entries)
    kinds = Counter(e.get("kind", "unknown") for e in entries)
    n = max(len(entries), 1)
    report = {
        "lines": len(entries),
        "char_share": {
            "thai": round(thai / total, 3),
            "latin": round(latin / total, 3),
            "digits": round(digit / total, 3),
        },
        "length": {
            "p50": lengths[len(lengths) // 2],
            "p95": lengths[int(len(lengths) * 0.95)],
            "max": lengths[-1],
            "over_max_len": sum(x > MAX_LEN for x in lengths),
        },
        "content_mix": {k: round(v / n, 3) for k, v in kinds.most_common()},
        "font_styles": {k: round(v / n, 3) for k, v in styles.most_common()},
        "chars_not_in_model_dictionary": dict(sorted(oov.items(), key=lambda x: -x[1])[:20]),
        "distinct_chars": len(chars),
    }
    loopless = report["font_styles"].get("loopless", 0.0)
    report["fit_checks"] = {
        "all characters readable by the model (in dictionary)": "PASS"
        if not oov
        else f"FAIL ({len(oov)} chars)",
        "no line longer than 25 characters": "PASS"
        if report["length"]["over_max_len"] == 0
        else "FAIL",
        "Thai is the main script (>= 50% of characters)": "PASS" if thai / total >= 0.5 else "WARN",
        "enough Latin for names/carriers (>= 10%)": "PASS" if latin / total >= 0.10 else "WARN",
        "enough digits for units/phones/tracking (>= 10%)": "PASS"
        if digit / total >= 0.10
        else "WARN",
        "loopless fonts well represented (>= 30% of lines)": (
            "PASS"
            if loopless >= 0.30
            else ("n/a (unknown fonts)" if "unknown" in styles else "WARN")
        ),
    }
    return report


def sample_sheet(rows: list[tuple[Image.Image, str]], path: Path, width: int = 900) -> None:
    sheet_h = sum(min(img.height, 70) + 6 for img, _ in rows)
    sheet = Image.new("RGB", (width, sheet_h), (255, 255, 255))
    y = 0
    for img, _ in rows:
        scale = min(1.0, 70 / img.height, (width - 10) / img.width)
        im = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
        sheet.paste(im, (5, y))
        y += im.height + 6
    sheet.save(path, quality=90)


def load_dictionary(cache: Path) -> set[str]:
    path = cache / "ppocrv5_th_dict.txt"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DICT_URL, path)
    return set(path.read_text(encoding="utf-8").splitlines())


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("data/thai_rec_synth"))
    ap.add_argument("--n", type=int, default=100_000, help="total lines (train+val+test)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "thai_rec_fonts")
    ap.add_argument("--evaluate", type=Path, help="only evaluate an existing PaddleOCR label file")
    args = ap.parse_args()
    dictionary = load_dictionary(args.cache)

    if args.evaluate:
        entries = [
            {"text": ln.split("\t", 1)[1]}
            for ln in args.evaluate.read_text(encoding="utf-8").splitlines()
            if "\t" in ln
        ]
        print(json.dumps(evaluate(entries, dictionary), indent=2, ensure_ascii=False))
        return

    if not features.check_feature("raqm"):
        raise SystemExit(
            "Pillow has no raqm layout: Thai marks would be placed wrongly. "
            "Install libraqm0 + libfribidi0 (Linux) and reinstall Pillow."
        )
    rng = random.Random(args.seed)
    corpus = Corpus(rng)
    fonts = download_fonts(args.cache)
    cmap = {p: font_chars(p) for p in fonts}
    test_fonts = {
        p
        for p in fonts
        if any(p.name == Path(t).name.replace("%5B", "[").replace("%5D", "]") for t in TEST_FONTS)
    }
    train_fonts = [p for p in fonts if p not in test_fonts]
    print(f"{len(train_fonts)} train fonts, {len(test_fonts)} held-out test fonts")

    img_dir = args.out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    kinds, weights = list(MIX), list(MIX.values())
    splits = {"train": [], "val": [], "test": []}
    entries, dropped, sheet_rows = [], Counter(), []
    for i in range(args.n):
        split = "test" if i % 20 == 0 else ("val" if i % 50 == 1 else "train")  # 5% test, 2% val
        kind = rng.choices(kinds, weights)[0]
        text = corpus.sample(kind)
        text = unicodedata.normalize("NFC", text)
        if any(ch not in dictionary and ch != " " for ch in text):
            dropped["not in model dictionary"] += 1
            continue
        pool = sorted(test_fonts) if split == "test" else train_fonts
        usable = [f for f in pool if set(text.replace(" ", "")) <= cmap[f]]
        if not usable:
            dropped["no font has all glyphs"] += 1
            continue
        # Pick the style first so the many system fonts cannot crowd out loopless ones.
        styles = sorted({fonts[f] for f in usable})
        style = rng.choices(styles, [STYLE_WEIGHTS.get(s, 0.2) for s in styles])[0]
        font = rng.choice([f for f in usable if fonts[f] == style])
        img = render(text, font, rng)
        name = f"images/{i:07d}.jpg"
        img.save(args.out / name, quality=rng.randint(70, 95))
        splits[split].append(f"{name}\t{text}")
        entries.append({"text": text, "kind": kind, "style": fonts[font], "split": split})
        if len(sheet_rows) < 40 and rng.random() < 0.02:
            sheet_rows.append((img, text))
        if (i + 1) % 10_000 == 0:
            print(f"{i + 1}/{args.n} rendered", flush=True)

    for split, lines in splits.items():
        (args.out / f"{split}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = {
        "splits": {k: len(v) for k, v in splits.items()},
        "dropped": dict(dropped),
        "fonts": {
            "train": sorted(p.name for p in train_fonts),
            "test_held_out": sorted(p.name for p in test_fonts),
        },
        "train": evaluate([e for e in entries if e["split"] != "test"], dictionary),
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    sample_sheet(sheet_rows, args.out / "samples.jpg")
    print(json.dumps({k: report[k] for k in ("splits", "dropped")}, ensure_ascii=False))
    print(json.dumps(report["train"]["fit_checks"], indent=2, ensure_ascii=False))
    print(f"wrote {args.out}/train.txt, val.txt, test.txt, report.json, samples.jpg")


if __name__ == "__main__":
    main()
