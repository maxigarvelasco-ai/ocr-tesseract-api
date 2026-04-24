from fastapi import FastAPI, UploadFile, File, HTTPException
from paddleocr import PaddleOCR
from PIL import Image, ImageOps, ImageEnhance
import tempfile
import os
import io
import re
import json

app = FastAPI()

ocr_engine = PaddleOCR(
    lang="en",
    ocr_version="PP-OCRv5",
    device="cpu",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    text_det_limit_side_len=1600,
    text_det_limit_type="max",
    text_rec_score_thresh=0.0,
)

@app.get("/")
def health():
    return {"status": "ok", "service": "paddleocr-api"}

def clean_text(text: str) -> str:
    text = text.replace("\x0c", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def preprocess_image(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")

    # Si la imagen es muy chica, la agrandamos. Instagram comprime mucho.
    w, h = image.size
    max_side = max(w, h)
    if max_side < 1800:
        scale = 1800 / max_side
        image = image.resize((int(w * scale), int(h * scale)))

    # Mejor contraste para flyers.
    image = ImageEnhance.Contrast(image).enhance(1.35)
    image = ImageEnhance.Sharpness(image).enhance(1.25)

    return image

def extract_rec_texts(result_obj):
    """
    PaddleOCR v3 devuelve objetos Result.
    Internamente suelen tener rec_texts dentro de res/json.
    Esta función intenta extraer textos sin depender demasiado del formato exacto.
    """
    texts = []

    def walk(obj):
        if obj is None:
            return

        if isinstance(obj, dict):
            if "rec_texts" in obj and isinstance(obj["rec_texts"], list):
                for t in obj["rec_texts"]:
                    if isinstance(t, str) and t.strip():
                        texts.append(t.strip())

            # A veces puede venir como text / label
            for key in ["text", "label"]:
                val = obj.get(key)
                if isinstance(val, str) and val.strip():
                    texts.append(val.strip())

            for v in obj.values():
                walk(v)

        elif isinstance(obj, list):
            for item in obj:
                walk(item)

        elif hasattr(obj, "__dict__"):
            walk(vars(obj))

    # Intento 1: objeto como dict interno
    try:
        walk(result_obj)
    except Exception:
        pass

    # Intento 2: algunos Result tienen atributo json/res
    for attr in ["json", "res"]:
        try:
            val = getattr(result_obj, attr, None)
            walk(val)
        except Exception:
            pass

    # Limpieza de duplicados manteniendo orden
    unique = []
    seen = set()
    for t in texts:
        t = clean_text(t)
        if t and t not in seen:
            seen.add(t)
            unique.append(t)

    return unique

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    temp_path = None

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image = preprocess_image(image)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            temp_path = tmp.name
            image.save(temp_path, format="JPEG", quality=95)

        results = ocr_engine.predict(temp_path)

        all_lines = []
        raw_debug = []

        for res in results:
            lines = extract_rec_texts(res)
            all_lines.extend(lines)

            # Debug liviano para ver qué devuelve Paddle si hace falta
            try:
                raw_debug.append(str(res)[:1500])
            except Exception:
                pass

        text = clean_text("\n".join(all_lines))

        return {
            "text": text,
            "engine_used": "paddleocr",
            "lines_count": len(all_lines),
            "lines": all_lines,
            "debug_sample": raw_debug[:1]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except Exception:
                pass