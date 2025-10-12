from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import gradio as gr
import re
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText, \
    PayloadSchemaType, TextIndexParams, TokenizerType, PointStruct

# Load wiki retrieval settings from Qdrant
print("Loading Qdrant client and embedding model...")
CLUSTER_URL = "https://2a834546-f9a5-4f6d-836a-11b02865d9d8.europe-west3-0.gcp.cloud.qdrant.io"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.878Wi2gnA9e2TRTOz7efjZddEGBFh2rI7eM76L_29e8"

client = QdrantClient(url=CLUSTER_URL, api_key=API_KEY, timeout=5.0)

embedder = SentenceTransformer('all-MiniLM-L6-v2')

collection_name = "wiki_medical"

# Create/check payload index for "title" as TEXT

title_index_created = False
try:
    if not client.collection_exists(collection_name):
        raise Exception(f"Collection '{collection_name}' not found! Run preprocessing first.")

    coll_info = client.get_collection(collection_name)
    payload_schema = getattr(coll_info, 'payload_schema', {}) or {}

    title_index_exists = False
    if 'title' in payload_schema:
        field_info = payload_schema['title']
        try:
            title_index_exists = (getattr(field_info, 'data_type', None) == PayloadSchemaType.TEXT) \
                                 or (getattr(field_info, 'type', None) == "text")
        except Exception:
            title_index_exists = False

    if not title_index_exists:
        try:
            client.delete_payload_index(collection_name=collection_name, field_name="title")
            print("Deleted existing payload index for 'title'.")
        except Exception:
            pass
        client.create_payload_index(
            collection_name=collection_name,
            field_name="title",
            field_schema=TextIndexParams(
                type=PayloadSchemaType.TEXT,
                tokenizer=TokenizerType.WORD,
                min_token_len=1,
                max_token_len=255,
                lowercase=True
            )
        )
        print(
            "Created TEXT payload index for 'title' with lowercase tokenizer! (Case-insensitive token matching ready.)")
    else:
        print("TEXT payload index for 'title' already exists. (Filter ready.)")

    title_index_created = True
except Exception as e:
    print(f"ERROR with payload index: {e}. Falling back to vector search without title filter.")
    title_index_created = False

# Load Llama 3 8B (4-bit)

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
print("Model loaded!")

STOP_WORDS = {
    'how', 'to', 'what', 'is', 'the', 'a', 'an', 'and', 'or', 'in', 'on',
    'for', 'with', 'are', 'of', 'by', 'from', 'about', 'into', 'than',
    'that', 'this', 'these', 'those', 'it', 'its', 'as', 'at', 'be', '?'
}


def normalize_text(s: str) -> str:
    return re.sub(r'[^a-z0-9 ]+', '', s.lower()).strip()


@lru_cache(maxsize=1024)
def _cached_embed(text: str):
    return embedder.encode([text])[0].tolist()


def batch_embed(texts):
    return embedder.encode(texts, batch_size=32, convert_to_numpy=True)


def generate_followup_queries(original_query, current_context, num_queries=2, previous_queries=None):
    if len(current_context) > 4000:
        current_context = current_context[:4000] + "\n[Truncated for brevity]"

    words = re.findall(r'\b[a-zA-Z]{4,}\b', original_query.lower())  # Only words >3 chars
    main_topic = next((w for w in words if w not in ['what', 'how', 'treat', 'poisoning']), 'condition')

    prev_str = ""
    if previous_queries:
        prev_str = f"Previous follow-ups (do not repeat): {', '.join(previous_queries)}\n"

    # focused on specified sections
    example = (
        f"Example 1 (for query 'Treatment for {main_topic}'):\n"
        f"1. What are the treatment and prevention options for {main_topic}?\n"
        f"2. What are the signs and symptoms of {main_topic}?\n\n"
        f"Example 2 (for query 'Causes of {main_topic}'):\n"
        f"1. What are the causes and pathophysiology of {main_topic}?\n"
        f"2. How is {main_topic} diagnosed?\n\n"
    )

    prompt = (
        f"You are a medical researcher. {example}"
        f"Generate EXACTLY {num_queries} NEW, DISTINCT follow-up questions focused ONLY on these sections: Definition/Introduction, Signs and Symptoms, Causes/Pathophysiology, Diagnosis, Treatment/Prevention.\n"
        "Use fallback-style questions like: 'What is the [section] of {main_topic}?', 'What are [key aspects] for {main_topic} in [section]?'\n"
        "Step 1: Reason briefly (1 sentence) on what info is missing from these sections.\n"
        "Step 2: Output ONLY numbered lines (no reason in output):\n"
        "1. [Specific question on one section?] \n2. [Specific question on another section?]\n"
        f"Original: {original_query}\n{prev_str}"
        f"Context so far:\n{current_context}\n\nFollow-up Questions:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            no_repeat_ngram_size=4,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    cands = []
    for ln in lines:
        m = re.match(r'^\s*\d+\.\s*(.+)\s*\?*\s*$', ln)
        if m:
            cand = m.group(1).strip() + '?'

            clean = re.sub(r'[^a-zA-Z0-9\s]', '', cand)
            if len(cand) > 10 and len(clean.split()) > 2:  #
                cands.append(cand)

    # Fallback cases
    fallback_base = [
        f"What is the definition and introduction to {main_topic}?",
        f"What are the signs and symptoms of {main_topic}?",
        f"What are the causes and pathophysiology of {main_topic}?",
        f"How is {main_topic} diagnosed?",
        f"What are the treatment and prevention options for {main_topic}?",
    ]
    if len(cands) < num_queries // 2:
        if previous_queries:
            fallback_base = [q for q in fallback_base if
                             normalize_text(q) not in [normalize_text(pq) for pq in previous_queries]]
        cands = fallback_base[:num_queries]

    seen, out = set(), []
    for q in cands + fallback_base:  # Append fallback if needed
        nq = normalize_text(q)
        if nq in seen or q in (previous_queries or []):
            continue
        seen.add(nq)
        out.append(q)
        if len(out) == num_queries:
            break

    print("Generated follow-up queries:", out)
    return out


def retrieve_chunks_fast(query: str, query_vec, k=10, seen_keys_global=None):
    if seen_keys_global is None:
        seen_keys_global = set()

    words = query.lower().split()
    title_keywords = list({w.strip('?') for w in words if w not in STOP_WORDS and len(w) > 2})[:5]
    qdrant_filter = None
    if title_index_created and title_keywords:
        text_conds = [FieldCondition(key="title", match=MatchText(text=kw)) for kw in title_keywords]
        qdrant_filter = Filter(should=text_conds)
        print(f"Applying TEXT filter with keywords: {title_keywords}")

    def _search(filter_obj):
        try:
            results = client.query_points(
                collection_name=collection_name,
                query=query_vec,
                limit=k,
                with_payload=True,
                query_filter=filter_obj
            )
            return results.points
        except Exception as e:
            print(f"query_points() failed ({'with' if filter_obj else 'without'} filter): {e}")
            return []

    hits = _search(qdrant_filter)
    # Fallback
    if (not hits or len(hits) < max(1, k // 4)) and qdrant_filter is not None:
        print("Few hits with filter. Retrying without filter...")
        hits = _search(None)

    unique = []
    for h in hits:
        p = h.payload or {}
        t = p.get('title')
        ci = p.get('chunk_index')
        if t is not None and ci is not None:
            key = (t, ci)
        else:
            key = (t, normalize_text(p.get('chunk', ''))[:80])
        # dedup global cross-iter
        if key not in seen_keys_global:
            seen_keys_global.add(key)
            unique.append(h)

    # Build context
    retrieved_chunks = [h.payload['chunk'] for h in unique if h.payload and 'chunk' in h.payload]
    retrieved_titles = [h.payload['title'] for h in unique if h.payload and 'title' in h.payload]

    cap = min(4, len(retrieved_chunks))
    parts = []
    for title, chunk in zip(retrieved_titles[:cap], retrieved_chunks[:cap]):
        parts.append(f"Term: {title}\n{chunk}")
    context = "\n\n".join(parts)

    # Top titles
    titles_short, seen_t = [], set()
    for t in retrieved_titles:
        if t not in seen_t:
            seen_t.add(t)
            titles_short.append(t)
            if len(titles_short) == 3:
                break

    print(f"Retrieved {len(unique)} unique hits (context length: {len(context)} chars)")
    if titles_short:
        print(f"Top titles: {titles_short}")

    return context, titles_short, seen_keys_global


# Main iterative RAG
def i_med_rag_query(original_query, num_iterations=2, num_queries_per_iter=2, k=10,
                    max_total_chunks=8):
    context_snippets = []
    seen_keys_global = set()
    previous_queries = []

    print(f"Starting i-MedRAG with {num_iterations} iterations...")

    last_iter_titles = ["No terms retrieved."]

    print("\n--- Initial Retrieval for Original Query ---")
    orig_emb = batch_embed([original_query])[0]
    orig_context, orig_titles, seen_keys_global = retrieve_chunks_fast(original_query, orig_emb, k=k,
                                                                       seen_keys_global=seen_keys_global)
    if orig_context:
        new_snippets = orig_context.split("\n\n")
        context_snippets.extend(new_snippets)
        if len(context_snippets) > max_total_chunks:
            context_snippets = context_snippets[-max_total_chunks:]
        last_iter_titles = orig_titles[:3] if orig_titles else ["No terms retrieved."]
        print(f"Initial accumulated snippets: {len(context_snippets)} (cap={max_total_chunks})")

    for iteration in range(1, num_iterations + 1):
        print(f"\n--- Iteration {iteration} ---")

        short_ctx_for_gen = "\n\n".join(context_snippets[-4:])
        followups = generate_followup_queries(original_query, short_ctx_for_gen,
                                              num_queries=num_queries_per_iter,
                                              previous_queries=previous_queries)

        # 1) embed batch for all follow-ups of the iteration
        embs = batch_embed(followups)

        # 2) search Qdrant in parallel (reduce network latency)
        def _search_one(args):
            fq, vec = args
            # vec needs list/np list for qdrant
            return retrieve_chunks_fast(fq, vec.tolist(), k=k, seen_keys_global=seen_keys_global)

        with ThreadPoolExecutor(max_workers=min(4, len(followups))) as ex:
            results = list(ex.map(_search_one, zip(followups, embs)))

        # 3) combine context of this iteration
        iteration_titles = []
        new_snippets = []
        for (fq_context, fq_titles, _seen) in results:
            if fq_context:
                new_snippets.extend(fq_context.split("\n\n"))
            if fq_titles:
                iteration_titles.extend(fq_titles)

        # Update list of follow-ups already asked to avoid repetition
        previous_queries.extend(followups)

        # 4) cap total accumulated snippets so prompt doesn't bloat
        if new_snippets:
            context_snippets.extend(new_snippets)
            if len(context_snippets) > max_total_chunks:
                context_snippets = context_snippets[-max_total_chunks:]

        last_iter_titles = iteration_titles[:3] if iteration_titles else ["No terms retrieved."]
        print(f"Accumulated snippets: {len(context_snippets)} (cap={max_total_chunks})")

    # Gen response

    if not context_snippets:
        prompt = f"""You are a medical expert. Provide a direct, detailed explanation with medical reasoning.
Query: {original_query}

Response:"""
    else:
        accumulated_context = "\n\n".join(context_snippets)
        prompt = (
            "Based on the following accumulated medical knowledge, answer the question directly with clinical reasoning. "
            "Be concise but specific. No meta-commentary or citations.\n\n"
            f"{accumulated_context}\n\n"
            f"Query: {original_query}\n\nResponse:"
        )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Response:" in response:
        response = response.split("Response:", 1)[-1].strip()

    # Post-processing
    meta_patterns = [r'This response provides.*', r'The answer is.*', r'In summary.*']
    for pattern in meta_patterns:
        response = re.sub(pattern, '', response, flags=re.IGNORECASE | re.DOTALL)
    response = re.sub(r'\[\d+\]', '', response)
    response = re.sub(r'\n\s*\n\s*\n', '\n\n', response.strip())

    return response, f"Retrieved terms (last iter): {', '.join(last_iter_titles)}"


# Demo UI

def demo_interface(query):
    if not query.strip():
        return "Please enter a query.", "No terms retrieved."
    response, titles = i_med_rag_query(
        original_query=query,
        num_iterations=2,
        num_queries_per_iter=2,
        k=10,
        max_total_chunks=10
    )
    debug = f"Retrieved terms (last iteration): {titles}"
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
    title="i-MedRAG Medical Demo (Fixed: No Garbage Queries)",
    description="Medical query → Iterative retrieval from Qdrant (batch follow-ups) → Generate reasoning (fast, relevant).",
    flagging_mode="never"
)

if __name__ == "__main__":
    iface.launch(share=False, server_name="127.0.0.1")