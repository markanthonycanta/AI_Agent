import logging
import chromadb
import PyPDF2
import docx
import os
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
import google.generativeai as genai

# Suppress extra logging messages from ChromaDB
logging.getLogger("chromadb").setLevel(logging.ERROR)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "AI Agent is running!"}

# ----------------------------
# Approach #2: Read each credential field from separate environment variables
# ----------------------------

creds_dict = {
    "type": os.getenv("type"),
    "project_id": os.getenv("project_id"),
    "private_key_id": os.getenv("private_key_id"),
    # Convert the escaped newlines "\\n" to actual newlines "\n"
    "private_key": os.getenv("private_key", "").replace("\\n", "\n"),
    "client_email": os.getenv("client_email"),
    "client_id": os.getenv("client_id"),
    "auth_uri": os.getenv("auth_uri"),
    "token_uri": os.getenv("token_uri"),
    "auth_provider_x509_cert_url": os.getenv("auth_provider_x509_cert_url"),
    "client_x509_cert_url": os.getenv("client_x509_cert_url"),
    "universe_domain": os.getenv("universe_domain"),
}

# Quick sanity check for the private key (usually the trickiest part)
if not creds_dict["private_key"]:
    raise ValueError("Missing or empty 'private_key' environment variable.")

print("Loaded service account info from separate environment variables.")

# Create credentials object
creds = Credentials.from_service_account_info(creds_dict, scopes=['https://www.googleapis.com/auth/drive'])
print("Google Drive credentials loaded successfully!")

# Initialize Google Drive service
drive_service = build('drive', 'v3', credentials=creds)
print("Google Drive service initialized.")

# ----------------------------
# ChromaDB Configuration
# ----------------------------
chroma_client = chromadb.PersistentClient(path="chroma_storage")
document_collection = chroma_client.get_or_create_collection("documents")
metadata_collection = chroma_client.get_or_create_collection("metadata")

# ----------------------------
# Gemini API Configuration
# ----------------------------
GEMINI_API_KEY = "AIzaSyAp4kj7syRvuC4_32iaUEJ2iSv53GmT42E"  # Replace with your actual API key
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# ----------------------------
# Request Model
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

# def chat_with_ai(user_input):
#     context_chunks = search_documents(user_input, n_results=3)
#     if context_chunks and any(chunk.strip() for chunk in context_chunks):
#         context = "\n---\n".join(context_chunks)
#         prompt = f"Use this context to answer:\n{context}\n\nQuestion: {user_input}\n\nAnswer:"
#     else:
#         prompt = f"Answer based on general knowledge:\n\nQuestion: {user_input}\n\nAnswer:"
#     return query_gemini(prompt)

last_question = None

def chat_with_ai(user_input):
    global last_question
    # List of confirmation responses (case-insensitive)
    confirmation_responses = ["yes", "yeah", "okay", "ok", "sure", "yup"]

    # If the input is a confirmation for a pending question, answer based on general knowledge.
    if last_question is not None and user_input.lower().strip() in confirmation_responses:
        prompt = (
            f"Answer based on general knowledge:\n\n"
            f"Question: {last_question}\n\n"
            f"Answer:"
        )
        answer = query_gemini(prompt)
        last_question = None  # Clear pending question after answering.
        return answer

    # Retrieve context chunks from ChromaDB.
    context_chunks = search_documents(user_input, n_results=3)
    if context_chunks and any(chunk.strip() for chunk in context_chunks):
        context = "\n---\n".join(context_chunks)
        prompt = (
            f"Use the following context to answer the question. "
            f"If the context lacks sufficient detail, supplement your answer using your own knowledge.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {user_input}\n\n"
            f"Answer:"
        )
        answer = query_gemini(prompt)
        # Check if the answer is unsatisfactory.
        if ("does not contain any information" in answer.lower() or len(answer.split()) < 5):
            last_question = user_input
            return ("The provided text does not contain sufficient information to answer your question. "
                    "Do you want me to answer based on my general knowledge?")
        else:
            return answer
    else:
        # No relevant context found; ask for confirmation.
        last_question = user_input
        return "No relevant context found. Do you want me to answer based on my general knowledge?"

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
