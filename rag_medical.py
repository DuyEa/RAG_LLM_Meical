from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import gradio as gr
import re  # For post-processing
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchAny, MatchValue, \
    PayloadSchemaType  # For filtering by title + index creation

# Load wiki retrieval settings from Qdrant
print("Loading Qdrant client and embedding model...")
CLUSTER_URL = "https://2a834546-f9a5-4f6d-836a-11b02865d9d8.europe-west3-0.gcp.cloud.qdrant.io"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.878Wi2gnA9e2TRTOz7efjZddEGBFh2rI7eM76L_29e8"
client = QdrantClient(url=CLUSTER_URL, api_key=API_KEY)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# FIX: Create/check payload index for "title"
collection_name = "wiki_medical"
title_index_created = False
try:
    if not client.collection_exists(collection_name):
        raise Exception(f"Collection '{collection_name}' not found! Run preprocessing first.")
    coll_info = client.get_collection(collection_name)
    payload_schema = getattr(coll_info, 'payload_schema', {})
    title_index_exists = 'title' in payload_schema and payload_schema['title'].data_type == PayloadSchemaType.KEYWORD
    if not title_index_exists:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="title",
            field_schema=PayloadSchemaType.KEYWORD
        )
        title_index_created = True
        print("Created payload index for 'title' field! (Filter will now work.)")
    else:
        title_index_created = True
        print("Payload index for 'title' already exists. (Filter ready.)")
except Exception as e:
    print(f"ERROR with payload index: {e}. Falling back to vector search without title filter.")
    title_index_created = False

# Load Llama 3 8B
print("Loading Llama 3 model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_enable_fp32_cpu_offload=True,
)
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
max_memory = {0: "5GiB", "cpu": "20GiB"}
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True,
    max_memory=max_memory,
    offload_folder=None,
    dtype=torch.float16,
    trust_remote_code=True,
)
print("Model loaded! RAG is ready.")


# RAG function with improved hybrid retrieval
def rag_query(query, k=10):  # Increase k for more context
    words = query.lower().split()
    stop_words = {'how', 'to', 'what', 'is', 'the', 'a', 'an', 'and', 'or', 'in', 'on', 'for', 'with', '?'}
    title_keywords = [word for word in words if word not in stop_words and len(word) > 2]

    # Try exact match first if possible
    exact_hits = []
    if title_index_created:
        exact_filter = Filter(must=[FieldCondition(key="title", match=MatchValue(value=query.strip('?').title()))])
        try:
            exact_emb = embedder.encode([query])[0].tolist()
            exact_hits = client.search(
                collection_name=collection_name,
                query_vector=exact_emb,
                limit=1,
                with_payload=True,
                query_filter=exact_filter
            )
        except:
            pass  # If exact fails, proceed to semantic

    # Semantic search with keyword filter
    qdrant_filter = None
    if title_index_created and title_keywords:
        qdrant_filter = Filter(must=[FieldCondition(key="title", match=MatchAny(any=title_keywords))])
        print(f"Applying title filter with keywords: {title_keywords}")

    query_emb = embedder.encode([query])[0].tolist()
    hits = client.search(
        collection_name=collection_name,
        query_vector=query_emb,
        limit=k,
        with_payload=True,
        query_filter=qdrant_filter,
    )

    # Combine exact + semantic, dedup
    all_hits = exact_hits + hits
    seen_ids = set()
    unique_hits = []
    for hit in all_hits:
        if hit.id not in seen_ids:
            seen_ids.add(hit.id)
            unique_hits.append(hit)

    retrieved_chunks = [hit.payload['chunk'] for hit in unique_hits]
    retrieved_titles = [hit.payload['title'] for hit in unique_hits]

    # Dedup titles, preserve order
    seen = set()
    unique_titles = [t for t in retrieved_titles if not (t in seen or seen.add(t))]

    context = "\n\n".join([f"Term: {title}\n{chunk}" for title, chunk in zip(retrieved_titles, retrieved_chunks)])
    # Enhanced prompt: Avoid meta-commentary and citations
    prompt = f"""Based on the following medical knowledge:
{context}
Focus on the exact query term and provide a direct, detailed definition or explanation with medical reasoning (step-by-step if possible). Do not add any meta-commentary, evaluation, summary about the response itself, or citations like [1], [2].
Query: {query}
Response:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            min_new_tokens=150,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("Response:")[-1].strip()

    # FIX: Post-process to remove meta-commentary and citations
    meta_patterns = [r'This response provides.*', r'The answer is.*', r'In summary.*']
    for pattern in meta_patterns:
        response = re.sub(pattern, '', response, flags=re.IGNORECASE | re.DOTALL)
    # Remove citations like [1], [2]
    response = re.sub(r'\[\d+\]', '', response)
    response = re.sub(r'\n\s*\n\s*\n', '\n\n', response.strip())  # Clean extra newlines

    return response, unique_titles[:3]


# Demo UI
def demo_interface(query):
    if not query.strip():
        return "Please enter a query.", "No terms retrieved."
    response, titles = rag_query(query)
    debug = f"Retrieved terms: {', '.join(titles)}"
    return response, debug


iface = gr.Interface(
    fn=demo_interface,
    inputs=gr.Textbox(
        label="Medical Query (e.g., 'What is paracetamol poisoning?')",
        placeholder="Enter query..."
    ),
    outputs=[
        gr.Textbox(label="Response", lines=16, show_copy_button=True),
        gr.Textbox(label="Retrieved Terms", lines=3)
    ],
    title="Medical RAG Demo (Wiki Retrieval + Llama 3 Generation)",
    description="Medical query → Retrieve wiki chunks from Qdrant → Generate reasoning.",
    flagging_mode="never"
)

if __name__ == "__main__":
    iface.launch(share=False, server_name="127.0.0.1")