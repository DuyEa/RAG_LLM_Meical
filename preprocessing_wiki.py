from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import uuid  # To create a unique ID for each chunk

# Load dataset
dataset_full = load_dataset("gamino/wiki_medical_terms")['train']
print(f"Full dataset size: {len(dataset_full)}")

# Take the first 1000 rows as a subset
dataset = dataset_full.select(range(1000))
print(f"Subset size: {len(dataset)}")

# Extract texts with metadata (title + text), and de-duplicate by title immediately
texts_with_meta = []
seen_titles = set()
for item in dataset:
    title = item.get('page_title', '').lower()
    text = item.get('page_text', '').strip()
    if title and text and title not in seen_titles:
        full_entry = f"Term: {title}\n{text}"
        texts_with_meta.append({'title': title, 'text': full_entry, 'index': len(texts_with_meta)})
        seen_titles.add(title)
print(f"Valid unique entries after de-dup: {len(texts_with_meta)}")

# Only use the text field for chunking
texts = [meta['text'] for meta in texts_with_meta]

# Chunk with larger size (500 chars)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
)
chunks_with_meta = []
for meta in texts_with_meta:
    entry_chunks = splitter.split_text(meta['text'])
    for i, chunk in enumerate(entry_chunks):
        chunks_with_meta.append({
            'chunk': chunk,
            'title': meta['title'],
            'entry_index': meta['index'],
            'chunk_index': i
        })
chunks = [meta['chunk'] for meta in chunks_with_meta]
print(f"Number of chunks: {len(chunks)}")
print("Sample chunk 0:", chunks[0])

# Embedding
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(chunks, batch_size=32, show_progress_bar=True)

# Connect to Qdrant (cloud - REPLACE with your URL and API_KEY)
CLUSTER_URL = "https://2a834546-f9a5-4f6d-836a-11b02865d9d8.europe-west3-0.gcp.cloud.qdrant.io"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.878Wi2gnA9e2TRTOz7efjZddEGBFh2rI7eM76L_29e8"
client = QdrantClient(url=CLUSTER_URL, api_key=API_KEY)

# Create collection if it doesn't exist
collection_name = "wiki_medical"
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE)
    )
    print(f"Created collection '{collection_name}'")
else:
    print(f"Collection '{collection_name}' already exists")

#Upsert in batches to avoid timeouts (1000 points per batch)
batch_size = 1000
total_points = len(chunks_with_meta)
points_list = []  # Pre-build the full list of points
for i, (chunk_meta, embedding) in enumerate(zip(chunks_with_meta, embeddings)):
    point_id = str(uuid.uuid4())
    payload = {
        "chunk": chunk_meta['chunk'],
        "title": chunk_meta['title'],
        "entry_index": chunk_meta['entry_index'],
        "chunk_index": chunk_meta['chunk_index']
    }
    points_list.append(PointStruct(id=point_id, vector=embedding.tolist(), payload=payload))

print(f"Starting batched upsert of {total_points} points (batch_size={batch_size})...")
upserted_count = 0
for i in range(0, total_points, batch_size):
    batch_points = points_list[i:i + batch_size]
    batch_num = i // batch_size + 1
    try:
        client.upsert(collection_name=collection_name, points=batch_points)
        upserted_count += len(batch_points)

        if batch_num == 1 or batch_num % 10 == 0:
            print(f"Upserted batch {batch_num}: {len(batch_points)} points (total: {upserted_count}/{total_points})")
    except Exception as e:
        print(f"Error in batch {batch_num}: {e}. Retrying once...")
        try:
            client.upsert(collection_name=collection_name, points=batch_points)
            upserted_count += len(batch_points)
            if batch_num == 1 or batch_num % 10 == 0:
                print(f"Retry success for batch {batch_num}! (total: {upserted_count}/{total_points})")
        except Exception as e2:
            print(f"Failed retry for batch {batch_num}: {e2}. Skipping batch.")
            continue

print(f"Final: Upserted {upserted_count}/{total_points} points into Qdrant!")

# Quick search test
query_text = "diabetes symptoms"
query_embedding = embedder.encode([query_text])[0].tolist()
hits = client.search(
    collection_name=collection_name,
    query_vector=query_embedding,
    limit=5,
    with_payload=True
)
print("Top 5 matches:")
for hit in hits:
    print(f"Score: {hit.score:.3f} | Title: {hit.payload['title']} | Chunk: {hit.payload['chunk'][:100]}...")

print("Done! Check the cloud dashboard if using Qdrant Cloud.")
