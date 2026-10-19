"""ThaiOCRBench scoring.

`official_score` re-implements the benchmark's own eval code (scb-10x/ThaiOCRBench,
eval_scripts/eval.py, page_ocr_metric.py, vqa_metric.py) so our numbers sit on the same scale
as the paper's leaderboard:

  Text recognition                       → cn_vqa_evaluation: lowercase, drop spaces/newlines;
                                           answer contained in prediction = 1, else ANLS kept
                                           only if >= 0.5
  Fine-grained text recognition,
  Full-page OCR, Handwritten extraction  → BMFL = mean(BLEU, METEOR, token-set F1, 1 - NED),
                                           Thai tokenized with pythainlp newmm

`our_metrics` adds what matters for the parcel project: CER and exact match after the
project's own normalization (Thai digits, SARA AM, tone-mark order...).
"""

from __future__ import annotations

import ast
import re
import warnings
from functools import lru_cache

from rapidfuzz.distance import Levenshtein

from src.text.normalize import normalize_common

_THAI = re.compile(r"[฀-๿]")

BMFL_TASKS = {"Fine-grained text recognition", "Full-page OCR", "Handwritten content extraction"}
ANLS_TASKS = {"Text recognition"}
SUPPORTED_TASKS = BMFL_TASKS | ANLS_TASKS


@lru_cache(maxsize=1)
def _nltk():
    import nltk
    from nltk.metrics import f_measure
    from nltk.translate import meteor_score

    return nltk, f_measure, meteor_score


def _ned(pred: str, gt: str) -> float:
    """Edit distance normalised by the longer string (benchmark definition)."""
    longest = max(len(pred), len(gt))
    return Levenshtein.distance(pred, gt) / longest if longest else 0.0


def bmfl(pred: str, gt: str) -> dict[str, float]:
    if not pred:
        return {"bleu": 0.0, "meteor": 0.0, "f_measure": 0.0, "nls": 0.0, "bmfl": 0.0}
    nltk, f_measure, meteor_score = _nltk()
    # The benchmark tokenizes both strings with newmm when either contains Thai.
    if _THAI.search(gt) or _THAI.search(pred):
        from pythainlp.tokenize import word_tokenize

        ref, hyp = word_tokenize(gt, engine="newmm"), word_tokenize(pred, engine="newmm")
    else:
        ref, hyp = gt.split(), pred.split()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # nltk warns on zero n-gram overlap; score is still 0
        bleu = float(nltk.translate.bleu([ref], hyp))
        meteor = float(meteor_score.meteor_score([ref], hyp))
    f = f_measure(set(ref), set(hyp))
    f = float(f) if f is not None else 0.0  # None when a side is empty
    nls = 1.0 - _ned(pred, gt)
    return {
        "bleu": bleu,
        "meteor": meteor,
        "f_measure": f,
        "nls": nls,
        "bmfl": (bleu + meteor + f + nls) / 4,
    }


def _anls(pred: str, answer: str) -> float:
    value = 1.0 - _ned(pred, answer)
    return value if value >= 0.5 else 0.0


def text_recognition_score(pred: str, gt: str) -> float:
    """cn_vqa_evaluation from the benchmark, including its list-literal answer branch."""

    def clean(s: str) -> str:
        return s.lower().strip().replace("\n", " ").replace(" ", "")

    p = clean(pred)
    if gt.startswith("[") and gt.endswith("]"):
        try:
            answers = [str(a) for a in ast.literal_eval(gt)]
        except (ValueError, SyntaxError):
            answers = [gt]
        score = 0.0
        for a in answers:
            a = clean(a)
            if len(a.split(",")) < 4:
                if a in p:
                    score = 1.0
            else:
                score = max(score, _anls(p, a))
        return score
    a = clean(gt)
    return 1.0 if a in p else _anls(p, a)


def official_score(task: str, pred: str, gt: str) -> float:
    if task in ANLS_TASKS:
        return text_recognition_score(pred, gt)
    if task in BMFL_TASKS:
        return bmfl(pred, gt)["bmfl"]
    raise ValueError(f"task not supported by an OCR pipeline: {task}")


def our_metrics(pred: str, gt: str) -> dict[str, float]:
    """CER (by GT length) and exact match after project normalization, whitespace ignored."""
    p = re.sub(r"\s+", "", normalize_common(pred))
    g = re.sub(r"\s+", "", normalize_common(gt))
    cer = Levenshtein.distance(p, g) / len(g) if g else float(bool(p))
    return {"cer": cer, "exact": float(p == g)}
