import logging
import chromadb
import PyPDF2
import docx
import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
import google.generativeai as genai

# Suppress extra logging messages from ChromaDB
logging.getLogger("chromadb").setLevel(logging.ERROR)

# Initialize FastAPI app
app = FastAPI()

# ----------------------------
# Load Google Drive credentials from Railway environment variable
# Debugging print statement
print("Checking GOOGLE_DRIVE_CREDENTIALS environment variable...")

SERVICE_ACCOUNT_INFO = os.getenv("GOOGLE_DRIVE_CREDENTIALS")

if not SERVICE_ACCOUNT_INFO:
    raise ValueError("GOOGLE_DRIVE_CREDENTIALS not found in environment variables!")

# Fix newline issue in the private key
creds_dict = json.loads(SERVICE_ACCOUNT_INFO)
creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")

# Load credentials from modified JSON
creds = Credentials.from_service_account_info(creds_dict, scopes=['https://www.googleapis.com/auth/drive'])

print("Google Drive credentials loaded successfully!")
# ----------------------------
# ChromaDB Configuration
# ----------------------------
chroma_client = chromadb.PersistentClient(path="chroma_storage")
document_collection = chroma_client.get_or_create_collection("documents")
metadata_collection = chroma_client.get_or_create_collection("metadata")

# ----------------------------
# Gemini API Configuration
# ----------------------------
GEMINI_API_KEY = "AIzaSyAp4kj7syRvuC4_32iaUEJ2iSv53GmT42E"  # Replace with your API key
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# ----------------------------
# Request Models
# ----------------------------
class QueryRequest(BaseModel):
    user_input: str

# ----------------------------
# Helper Functions
# ----------------------------
def extract_text_from_pdf(file_path):
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text, chunk_size=500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def list_drive_files():
    results = drive_service.files().list(fields="files(id, name, mimeType)").execute()
    return results.get('files', [])

def download_drive_file(file_id, file_name, mime_type):
    export_mime_types = {
        "application/vnd.google-apps.document": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    }
    
    if mime_type.startswith("application/vnd.google-apps"):
        if mime_type in export_mime_types:
            export_type = export_mime_types[mime_type]
            request = drive_service.files().export_media(fileId=file_id, mimeType=export_type)
            file_name += ".docx"
        else:
            return None
    else:
        request = drive_service.files().get_media(fileId=file_id)

    with open(file_name, "wb") as file:
        file.write(request.execute())
    return file_name

def process_drive_files():
    files = list_drive_files()
    for file in files:
        file_id = file["id"]
        file_name = file["name"]
        mime_type = file.get("mimeType", "")
        
        processed = metadata_collection.get(ids=[file_id]).get("ids", [])
        if processed:
            print(f"Skipping already processed file: {file_name}")
            continue

        try:
            drive_service.files().get(fileId=file_id).execute()
        except Exception as e:
            print(f"Error accessing file {file_name}: {e}")
            continue
        
        downloaded_file = download_drive_file(file_id, file_name, mime_type)
        if not downloaded_file:
            print(f"Skipping unsupported file type: {file_name}")
            continue
        
        try:
            if downloaded_file.endswith(".pdf"):
                content = extract_text_from_pdf(downloaded_file)
            elif downloaded_file.endswith(".docx"):
                content = extract_text_from_docx(downloaded_file)
            elif downloaded_file.endswith(".txt"):
                content = extract_text_from_txt(downloaded_file)
            else:
                print(f"Skipping unsupported file type: {file_name}")
                continue
        except Exception as e:
            print(f"Error extracting text from {file_name}: {e}")
            continue
        
        chunks = chunk_text(content, chunk_size=500)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{file_id}_{i}"
            document_collection.add(ids=[chunk_id], documents=[chunk])
        
        metadata_collection.add(ids=[file_id], documents=[""], metadatas=[{"name": file_name}])
        print(f"Processed new file: {file_name}")

        os.remove(downloaded_file)

def search_documents(query, n_results=3):
    results = document_collection.query(query_texts=[query], n_results=n_results)
    return results.get("documents", [[]])[0]

def query_gemini(prompt):
    response = model.generate_content(prompt)
    return response.text

def chat_with_ai(user_input):
    context_chunks = search_documents(user_input, n_results=3)
    if context_chunks and any(chunk.strip() for chunk in context_chunks):
        context = "\n---\n".join(context_chunks)
        prompt = f"Use this context to answer:\n{context}\n\nQuestion: {user_input}\n\nAnswer:"
    else:
        prompt = f"Answer based on general knowledge:\n\nQuestion: {user_input}\n\nAnswer:"
    return query_gemini(prompt)

# ----------------------------
# FastAPI Endpoints
# ----------------------------
@app.get("/")
def root():
    return {"message": "AI Agent is running!"}

@app.post("/ask")
def ask_ai(request: QueryRequest):
    return {"response": chat_with_ai(request.user_input)}

# ----------------------------
# Startup Processing
# ----------------------------
@app.on_event("startup")
def startup_event():
    print("Processing Google Drive files...")
    process_drive_files()
