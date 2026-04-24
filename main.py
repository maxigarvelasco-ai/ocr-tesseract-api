from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image, ImageOps, ImageEnhance
import pytesseract
import io

app = FastAPI()

@app.get("/")
def health():
    return {"status": "ok", "service": "tesseract-ocr-api"}

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Preprocesamiento simple para flyers
        image = ImageOps.grayscale(image)
        image = ImageEnhance.Contrast(image).enhance(1.7)

        text = pytesseract.image_to_string(
            image,
            lang="spa",
            config="--oem 3 --psm 6"
        )

        return {
            "text": text.strip()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))