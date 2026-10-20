from pydantic import BaseModel


class Line(BaseModel):
    text: str
    score: float
    box: list[list[int]]  # four [x, y] corners, in the straightened waybill image


class Barcode(BaseModel):
    text: str
    format: str
    box: list[list[float]] | None = None  # four corners, in the straightened label image


class Waybill(BaseModel):
    found: bool
    quad: list[list[float]] | None = None  # corners in the uploaded photo
    detector: bool = False  # True when the trained detector was used


class OcrResponse(BaseModel):
    waybill: Waybill
    lines: list[Line]
    text: str
    barcodes: list[Barcode]
    timing_ms: dict[str, int]


class ReadBox(BaseModel):
    """Fractions (0-1) of the uploaded photo, as Parcel expects."""

    x: float
    y: float
    w: float
    h: float


class ReadField(BaseModel):
    value: str | int | None = None
    confidence: float = 0.0
    box: ReadBox | None = None


class ReadResponse(BaseModel):
    """The label read in Parcel's OCR contract (backend/app/domain/types.py OcrResult)."""

    fields: dict[str, ReadField]
    label_count: int
    quality_flags: list[str]
    model_version: str
