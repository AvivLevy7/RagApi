from fastapi import FastAPI, UploadFile, File
import pypdf
import io
from langchain_text_splitters import RecursiveCharacterTextSplitter

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "RAG API is running!"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # Read the file content and extract text
    contents = await file.read()
    pdf_reader = pypdf.PdfReader(io.BytesIO(contents))

    extracted_text = ""
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            extracted_text += text + "\n"

    # Configure the LangChain text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Size of each chunk (number of characters)
        chunk_overlap=200,  # Number of characters to overlap between chunks
        length_function=len  # Function to measure length (standard string length)
    )

    # Perform the actual splitting - returns a list of strings (chunks)
    chunks = text_splitter.split_text(extracted_text)

    # Print to the terminal for debugging/verification
    print(f"--- File split into {len(chunks)} chunks ---")
    if chunks:
        print("Preview of the first chunk:")
        # Slicing the string to reverse it [::-1] so Hebrew prints correctly in the terminal (LTR)
        print(chunks[0][:200][::-1])

        # Return a JSON response to the client (e.g., Swagger UI)
    return {
        "filename": file.filename,
        "status": "Success",
        "total_chunks_created": len(chunks)
    }