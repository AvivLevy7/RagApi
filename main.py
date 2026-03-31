
from fastapi import FastAPI, UploadFile, File
import pypdf
import io

app = FastAPI()

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    pdf_reader = pypdf.PdfReader(io.BytesIO(contents))

    extracted_text = ""
    for page in pdf_reader.pages:
        extracted_text += page.extract_text() + "\n"

    print(f"--- הטקסט שחולץ ---"[::-1])
    print(extracted_text[:500][::-1])

    return {"filename": file.filename, "status": "Success"}


