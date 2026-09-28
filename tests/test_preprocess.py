from PIL import Image, ImageDraw

from local_ocr.preprocess import prepare_scan


def test_prepare_scan_returns_rgb():
    image = Image.new("RGB", (80, 40), "white")
    ImageDraw.Draw(image).text((8, 12), "FATURA", fill="black")
    out = prepare_scan(image)
    assert out.mode == "RGB"
    assert out.size[0] >= 80
