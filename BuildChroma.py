import os
import json
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Path configuration
CHROMA_DB_PATH = "chroma_db"
DATA_FOLDER = "cancer_data"

# Load HuggingFace embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Text splitting : 500 characters per chunk with 50 character overlap
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Collect document chunks
docs = []

for filename in os.listdir(DATA_FOLDER):
    if filename.endswith(".json"):
        file_path = os.path.join(DATA_FOLDER, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                # If the JSON is a dict, concatenate its values
                if isinstance(data, dict):
                    content = "\n".join([str(v) for v in data.values()])
                elif isinstance(data, list):
                    content = "\n".join([str(item) for item in data])
                else:
                    content = str(data)
            except Exception as e:
                print(f"Failed to parse: {filename} - {e}")
                continue

            chunks = text_splitter.split_text(content)
            for chunk in chunks:
                docs.append(Document(page_content=chunk, metadata={"source": filename}))

# Build and persist the vector database
db = Chroma.from_documents(docs, embedding=embeddings, persist_directory=CHROMA_DB_PATH)
db.persist()

print(f"Successfully imported {len(docs)} text chunks into {CHROMA_DB_PATH}")
