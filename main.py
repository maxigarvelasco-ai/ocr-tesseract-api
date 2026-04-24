from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image, ImageOps, ImageEnhance
from rapidocr_onnxruntime import RapidOCR
import pytesseract
import numpy as np
import tempfile
import io
import re
import os

app = FastAPI()

rapid_engine = RapidOCR()

@app.get("/")
def health():
    return {"status": "ok", "service": "rapidocr+tesseract-api"}

def clean_text(text: str) -> str:
    text = text.replace("\x0c", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def run_tesseract(image: Image.Image) -> str:
    # Agrandar y mejorar contraste para flyers
    w, h = image.size
    image = image.resize((w * 3, h * 3))
    gray = ImageOps.grayscale(image)
    gray = ImageEnhance.Contrast(gray).enhance(2.2)
    gray = ImageEnhance.Sharpness(gray).enhance(2.0)

    configs = [
        "--oem 3 --psm 6",
        "--oem 3 --psm 11",
        "--oem 3 --psm 12",
    ]

    texts = []
    for config in configs:
        try:
            txt = pytesseract.image_to_string(gray, lang="spa+eng", config=config)
            txt = clean_text(txt)
            if txt:
                texts.append(txt)
        except Exception:
            pass

    return max(texts, key=len) if texts else ""

def run_rapidocr(image: Image.Image) -> str:
    # RapidOCR funciona mejor guardando temporalmente la imagen
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        temp_path = tmp.name
        image.convert("RGB").save(temp_path, format="JPEG", quality=95)

    try:
        result, _ = rapid_engine(temp_path)

        lines = []
        if result:
            for item in result:
                # item suele ser: [box, text, confidence]
                if len(item) >= 2:
                    text = str(item[1]).strip()
                    if text:
                        lines.append(text)

        return clean_text("\n".join(lines))

    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        rapid_text = run_rapidocr(image)
        tesseract_text = run_tesseract(image)

        # Elegimos el más largo, pero devolvemos ambos para debug
        best_text = rapid_text if len(rapid_text) >= len(tesseract_text) else tesseract_text

        return {
            "text": best_text,
            "engine_used": "rapidocr" if best_text == rapid_text else "tesseract",
            "rapid_text": rapid_text,
            "tesseract_text": tesseract_text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))