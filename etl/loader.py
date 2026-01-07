# etl/loader.py
import os
import requests
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from .parser import LoLPatchParser # Import our parser

load_dotenv()

# 1. Config
PINECONE_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "lol-nexus"
# Official Riot Patch Note URL (Example: 14.1)
PATCH_URL = "https://www.leagueoflegends.com/en-us/news/game-updates/patch-14-1-notes/"
PATCH_VERSION = "14.1"

# 2. Initialize Pinecone
pc = Pinecone(api_key=PINECONE_KEY)

# Check if index exists, if not create it
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating index: {INDEX_NAME}...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=384, # Matches the embedding model size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# 3. Setup Embeddings (Uses your HF Token automatically)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def main():
    print(f"Fetching Patch {PATCH_VERSION} from Riot...")
    response = requests.get(PATCH_URL)
    
    if response.status_code != 200:
        print("Failed to fetch website.")
        return

    # Parse
    parser = LoLPatchParser(response.text, PATCH_VERSION)
    chunks = parser.parse()
    print(f"Extracted {len(chunks)} data points.")

    if not chunks:
        print("No chunks found. Check the HTML parser logic.")
        return

    # Vectorize & Upload
    print("Embedding and uploading to Pinecone...")
    texts = [c.to_text() for c in chunks]
    metadatas = [{"patch": c.patch_version, "champ": c.header} for c in chunks]
    
    PineconeVectorStore.from_texts(
        texts=texts,
        embedding=embeddings,
        index_name=INDEX_NAME,
        metadatas=metadatas
    )
    print("Success! Data is in the DB.")

if __name__ == "__main__":
    main()