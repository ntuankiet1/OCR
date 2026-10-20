from pydantic import BaseModel


class Line(BaseModel):
    text: str
    score: float
    box: list[list[int]]  # four [x, y] corners, in the straightened waybill image


class Barcode(BaseModel):
    text: str
    format: str


class Waybill(BaseModel):
    found: bool
    quad: list[list[float]] | None = None  # corners in the uploaded photo


class OcrResponse(BaseModel):
    waybill: Waybill
    lines: list[Line]
    text: str
    barcodes: list[Barcode]
    timing_ms: dict[str, int]
