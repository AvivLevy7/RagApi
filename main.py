from fastapi import FastAPI, UploadFile, File
import pypdf
import io
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

app = FastAPI()

# Initialize the embedding model (runs locally, free of charge)
# The model "all-MiniLM-L6-v2" will be downloaded to your machine the first time you run this
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@app.get("/")
def read_root():
    return {"message": "RAG API is running!"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # 1. Read the file content and extract text
    contents = await file.read()
    pdf_reader = pypdf.PdfReader(io.BytesIO(contents))

    extracted_text = ""
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            extracted_text += text + "\n"

    # 2. Chunking - split text into manageable blocks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(extracted_text)

    # 3. Embeddings & Vector DB - The New Step
    # This converts our text chunks into numbers and saves them in a local folder called "chroma_db"
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    # Return success response
    return {
        "filename": file.filename,
        "status": "Success",
        "total_chunks_saved": len(chunks),
        "message": "Document embedded and successfully stored in local ChromaDB"
    }