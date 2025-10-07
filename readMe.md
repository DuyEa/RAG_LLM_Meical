# 🧠 RAG-LLM Medical QA Pipeline  

> **An end-to-end Retrieval-Augmented Generation (RAG) and Fine-Tuning pipeline** for building domain-specific medical question-answering models using **Llama-3-8B**, **LoRA**, and **Qdrant**.

---

## 📦 Project Overview
This repository demonstrates a **medical reasoning LLM pipeline** combining both **fine-tuning** and **retrieval-augmented generation**.  
Inspired by the 2024 study *“Development and Testing of Retrieval Augmented Generation in Large Language Models – A Case Study Report”*, the system integrates medical knowledge efficiently while managing computational cost.

### 🧩 Architecture
| Stage | File | Description |
|-------|------|-------------|
| **1. Dataset Preprocessing (QA)** | `preprocessing_qa.py` | Cleans and formats the “medical-o1-reasoning-SFT” dataset into reasoning-chain Q/A pairs. |
| **2. Knowledge Embedding (Wiki)** | `preprocessing_wiki.py` | Splits and embeds Wikipedia medical pages into vector form and uploads to **Qdrant** Cloud for retrieval. |
| **3. Fine-Tuning (LoRA)** | `finetune_medical.py` | Fine-tunes **Meta-Llama-3-8B** using 4-bit quantization and LoRA for reasoning tasks. |
| **4. Retrieval + Generation (RAG)** | `rag_medical.py` | Performs hybrid retrieval from Qdrant and contextual generation with the fine-tuned model, exposed through a **Gradio** demo UI. |

---

## ⚙️ Features
- **Fine-Tuning with LoRA + 4-bit quantization** (efficient for consumer GPUs like RTX 3060).  
- **Retrieval-Augmented Generation (RAG)** using **SentenceTransformers + Qdrant**.  
- **Medical-specific datasets** (`FreedomIntelligence/medical-o1-reasoning-SFT`, `gamino/wiki_medical_terms`).  
- **Gradio interface** for quick testing of medical queries.  
- **Secure token handling** (read from environment variables).

---

## 🚀 Setup & Usage

### 1️⃣ Environment
```bash
conda create -n rag_medical python=3.10
conda activate rag_medical
pip install torch transformers datasets sentence-transformers qdrant-client gradio peft trl bitsandbytes
```

### 2️⃣ Set Access Tokens
**PowerShell (Windows):**
```powershell
$env:HF_TOKEN="your_huggingface_token"
$env:QDRANT_API_KEY="your_qdrant_api_key"
```

**Linux/macOS:**
```bash
export HF_TOKEN="your_huggingface_token"
export QDRANT_API_KEY="your_qdrant_api_key"
```

---

### 3️⃣ Run Preprocessing
```bash
python preprocessing_qa.py
python preprocessing_wiki.py
```

This will:
- Clean & save QA dataset to `processed_medical_qa_fixed_en/`
- Upload 1000 embedded wiki chunks to your **Qdrant** cloud collection (`wiki_medical`)

---

### 4️⃣ Fine-Tune Model
```bash
python finetune_medical.py
```
**Notes:**
- Uses 4-bit quantization + LoRA (~5 GiB VRAM).  
- Each training batch takes **60–120 seconds** on RTX 3060.  
- Fine-tuning remains **significantly slower** than RAG inference.

---

### 5️⃣ Run RAG Demo
```bash
python rag_medical.py
```
Then open http://127.0.0.1:7860 to test.

Example queries:
```
What is diabetic ketoacidosis?
Explain paracetamol poisoning.
When should aspirin be stopped before surgery?
```

---

## 🧪 Performance Notes

| Operation | Typical Time (RTX 3060) |
|------------|-------------------------|
| Qdrant Search + Retrieval | **60–120 s** (depends on network & batch size) |
| Llama-3 Response Generation | **15–25 s** for ~200 tokens |
| One Fine-Tune Epoch (2k samples) | **~2 hours** |

---

## 🩺 Key Insights (from paper 2402.01733v1)
- **RAG > Fine-tuning alone** for clinical accuracy (91.4% vs. 80.1%).  
- Fine-tuning improves *consistency* but is **computationally expensive**.  
- RAG enhances *grounded factuality* and reduces hallucination.  
- A hybrid approach balances reasoning quality with feasible compute cost.

---

## 🧰 Repository Structure
```
RAG_LLM/
├── preprocessing_qa.py         # Clean and format QA dataset
├── preprocessing_wiki.py       # Chunk + embed wiki data to Qdrant
├── finetune_medical.py         # LoRA fine-tuning (Llama 3)
├── rag_medical.py              # Retrieval-Augmented Generation demo
├── .gitignore
└── README.md                   # (this file)
```

---

## 📚 References
- Yu He Ke et al., *“Development and Testing of Retrieval Augmented Generation in Large Language Models – A Case Study Report”*, arXiv:2402.01733 (2024).  
- Hugging Face Transformers & TRL SFT.  
- Qdrant Cloud API Docs.  
