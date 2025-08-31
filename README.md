# AdvancedRAG

Plan:
Want to create Local MultiModel LLM based RAG application which takes PFD(contains text, images, list and tables) as input and reply based on my query.
1. Everything must be local, NO online stuff
2. Want to use Qdrant as Vector DB and Qdrant should be running as a Docker server
3. Want to use Ollama as Orchestration Framework and Ollama should be running as a Docker server
4. Application must have models can generate Embeddings, please provide your input ?
5. Application must have final LLM models, please provide your input ?
6. Application must have techniques line buffering and document splitting

First plan the application and share all the software components are required ?

---

# 🏗️ Application Plan: Local Multi-Model RAG with Qdrant + Ollama

## 1. **Core Architecture**

Your app will have 4 key layers:

1. **Document Ingestion**

   * Extract **text, tables, lists, and images (OCR)** from PDFs.
   * Split documents into smaller chunks (document splitting).
   * Store embeddings in Qdrant.

2. **Vector Store (Qdrant)**

   * Runs in **Docker container**.
   * Stores embeddings + metadata for retrieval.

3. **Orchestration (Ollama)**

   * Also runs in **Docker container**.
   * Manages multiple local models (embeddings + LLMs).
   * Handles prompt orchestration and RAG pipeline.

4. **Query Processing**

   * User query → embeddings → vector search in Qdrant → context retrieval → passed to final LLM → answer generated.

---

## 2. **Software Components Required**

### 🔹 A. Document Processing Layer

* **Libraries:**

  * `pymupdf` or `pdfplumber` → extract **text + tables** from PDFs.
  * `unstructured` → robust parsing of text, lists, tables.
  * `pytesseract` + `opencv-python` → OCR for images in PDF.
  * `langchain-text-splitter` (or custom splitter) → document chunking & buffering.

---

### 🔹 B. Embedding Models (for Vector DB)

You need a **local embedding model** (fast + multilingual support). Options:

* **`nomic-embed-text`** → very strong open-source embedding model (available in Ollama).
* **`bge-m3`** (FlagEmbedding) → great for retrieval, supports multiple languages.
* **`sentence-transformers`** family (e.g. `all-MiniLM-L6-v2`) → lightweight, CPU-friendly.

👉 Embedding model must run **inside Ollama** or via **local inference engine** (e.g., HuggingFace).

---

### 🔹 C. Vector Database

* **Qdrant** (in Docker)

  * Stores embeddings.
  * Provides similarity search + filtering.
  * Exposes REST/gRPC API to your app.

---

### 🔹 D. Large Language Models (LLMs)

For final query answering, you want multiple **local LLMs** in Ollama. Options:

* **Mistral 7B Instruct** → fast + general reasoning.
* **LLaMA 3 8B Instruct** → strong performance, reasoning, alignment.
* **Qwen2 7B or 14B** → good for multilingual + reasoning.
* **Phi-4** (small, fast, efficient).

👉 You can orchestrate:

* **Small LLM** (fast, lightweight) for reranking / query rewriting.
* **Bigger LLM** (e.g. LLaMA3 8B, Mistral 7B) for final answer generation.

---

### 🔹 E. Orchestration Framework

* **Ollama** (in Docker) → runs multiple models locally.
* Your app (Python FastAPI or Flask backend) will:

  1. Accept PDF uploads.
  2. Process & store in Qdrant.
  3. Handle user queries → embedding → retrieval → context → pass to Ollama → generate final response.

---

### 🔹 F. Techniques for Better RAG

* **Document Splitting** → Break text into chunks (\~512–1000 tokens).
* **Buffering** → Store query history (so answers stay contextual).
* **Hybrid Retrieval (optional)** → Combine dense embeddings + keyword search.
* **Reranking** → Use a small LLM or embedding similarity to refine top results before passing to final LLM.

---

## 3. **Software Stack Summary**

### 🐳 **Dockerized Services**

* **Qdrant** → Vector DB.
* **Ollama** → LLM orchestration & embedding model hosting.

### 🐍 **Python Application**

* **FastAPI/Flask** → API server for your RAG app.
* **Libraries:**

  * `langchain` or `llama-index` (optional, to simplify RAG pipelines).
  * `pymupdf` / `pdfplumber` / `unstructured` → PDF parsing.
  * `pytesseract` + `opencv-python` → OCR.
  * `qdrant-client` → Connect to Qdrant.
  * `requests` → Call Ollama API.

### 🧠 **Models**

* Embedding Models: `nomic-embed-text`, `bge-m3`.
* LLMs: `Mistral-7B-Instruct`, `LLaMA3-8B`, `Phi-4`, `Qwen2`.

---

✅ This setup ensures:

* **Fully local** (no cloud calls).
* **Supports PDFs with text, tables, and images**.
* **Efficient retrieval with Qdrant**.
* **Multiple LLMs managed by Ollama**.

# 1️⃣ High-Level Application Structure

Your app has **3 main layers**:

1. **Data Ingestion & Processing** (PDF → Text/Images → Chunks)
2. **Knowledge Store** (Embeddings + Qdrant)
3. **Query & Response Generation** (User query → Retrieval → Ollama → Answer)

---

# 2️⃣ Suggested **Modular Architecture**

Here’s how I’d split it into Python modules/packages:

```
advanced_rag/
│── main.py                  # FastAPI entry point (routes & orchestration)
│── config.py                # Central config (paths, model names, Qdrant URL, etc.)
│
├── ingestion/               # PDF ingestion & preprocessing
│   ├── pdf_parser.py        # Extract text, tables, images from PDFs
│   ├── ocr.py               # OCR support for scanned images (pytesseract + opencv)
│   └── splitter.py          # Chunking & buffering logic
│
├── embeddings/              # Embedding logic
│   ├── embedder.py          # Call Ollama embedding model or local HuggingFace model
│
├── vectorstore/             # Qdrant vector DB operations
│   └── qdrant_handler.py    # Insert, search, delete, manage collections
│
├── llm/                     # LLM interaction
│   └── ollama_client.py     # Query Ollama for answer generation
│
├── retrieval/               # Retrieval pipeline
│   └── retriever.py         # Hybrid retrieval, reranking, context building
│
└── utils/                   # Utilities
    └── logger.py            # Logging setup, helpers
```

---

# 3️⃣ Module Responsibilities

### 📂 `ingestion/`

* **`pdf_parser.py`**

  * Extract raw text, tables, lists from PDFs (`pymupdf` / `pdfplumber` / `unstructured`).
* **`ocr.py`**

  * Extract text from images using `pytesseract` + `opencv`.
* **`splitter.py`**

  * Implement document chunking & buffering strategies (e.g., 500–1000 tokens).

---

### 📂 `embeddings/`

* **`embedder.py`**

  * Generate embeddings:

    * Option A: Call Ollama (`nomic-embed-text`, `bge-m3`).
    * Option B: Use local HuggingFace model (`sentence-transformers`).
  * Abstraction lets you switch between embedding backends.

---

### 📂 `vectorstore/`

* **`qdrant_handler.py`**

  * Manage collections (create, delete).
  * Insert embeddings with metadata.
  * Perform similarity search / filtering.

---

### 📂 `llm/`

* **`ollama_client.py`**

  * Send context + query to Ollama LLMs (Mistral, LLaMA3, etc.).
  * Handle streaming responses.
  * Could support multi-LLM orchestration (small model → rerank, big model → final answer).

---

### 📂 `retrieval/`

* **`retriever.py`**

  * Define RAG pipeline:

    1. Embed query.
    2. Retrieve top-N from Qdrant.
    3. Optionally rerank results.
    4. Build prompt for final LLM.

---

### 📂 `utils/`

* **`logger.py`**

  * Central logging.
* Can also add caching utilities, config loaders, etc.

---

# 4️⃣ Example Flow (How Modules Interact)

1. **Ingestion**

   * `pdf_parser.py` → extract text/tables/images.
   * `ocr.py` → OCR images.
   * `splitter.py` → chunk + buffer text.

2. **Embeddings + Storage**

   * `embedder.py` → generate embeddings.
   * `qdrant_handler.py` → store embeddings in Qdrant.

3. **Query Flow**

   * User query → `embedder.py` → vector search in `qdrant_handler.py`.
   * Retrieved chunks → `retriever.py` builds context.
   * `ollama_client.py` → final LLM generates answer.

---

✅ This design gives you:

* **Loose coupling** (easy to swap models or DB).
* **Extensibility** (e.g., add reranking later).
* **Maintainability** (each module has one responsibility).

---



Base LLM: LLaMA 3.1 8B Instruct (Q4_K_M under Ollama)
Embeddings: bge-m3
Vector DB: Qdrant
Orchestration Framework: LlamaIndex (proto) + LangChain (prod)

# Prerequsits
1. Create Linux distro OR VM on Linux or WSL on Windows to create a Linux runnable
2. Enable docker integration
3. Git pull to save our work
	- git clone https://github.com/dipeshgenai2025/BasicRAG

4. Open VS Code and sync with the working directory of Linux VM
5. Open terminal and create python environment
	- $ sudo apt-get update
	- $ sudo apt-get install python3-venv
	- $ sudo apt-get install python3-pip -y
  - $ sudo apt-get install tesseract-ocr -y
	- $ python3 -m venv .AdvancedRAG
	- $ source .AdvancedRAG/bin/activate

6. Install required Python packages
- $ pip3 install -r requirements.txt

7. Download the Image to text model and save offline
- git clone https://huggingface.co/Salesforce/blip-image-captioning-base into /AdvancedRAG/Models/ImageCaptionModels/blip
- git clone https://huggingface.co/sentence-transformers/all-mpnet-base-v2 into /AdvancedRAG/Models/EmbeddingModels/mpnet-base-v2

8. Launch the Qdrant docker image
- Please visit following link for Qdrant information,  https://qdrant.tech/documentation/quickstart/
- $ docker pull qdrant/qdrant
- $ docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v "$(pwd)/QdrantDBStorage:/qdrant/storage:z" qdrant/qdrant
- And for the next run just docker start qdrant

9. Launch the Ollama docker image
- $ docker pull ollama/ollama
- $ docker run -d --name ollama -p 11434:11434 -v $(pwd)/Models/MainLLM:/root/.ollama ollama/ollama:latest
- $ docker run -d --name ollama --gpus all -p 11434:11434 -v $(pwd)/Models/MainLLM:/root/.ollama ollama/ollama:latest

	This will start the ollama server and same can be checked on http://localhost:11434
- And for the next run just docker start ollama

10. Download Ollama comapatible AI models
- $ docker exec -it ollama ollama pull gemma3:4b
- $ docker exec -it ollama ollama pull mistral:7b
- $ docker exec -it ollama ollama pull llama3.1:8b
- $ docker exec -it ollama ollama pull deepseek-r1:8b









Prerequsits for Windows
1. WSL
2. Docker for Windows

1. Create Linux distro OR VM on Linux or WSL on Windows to create a Linux runnable
- Enable docker integration

2. Create seperate directory for this project
- $ mkdir SchoolStudyRAG
- $ cd SchoolStudyRAG

3. Create Python virtual environment
- $ sudo apt-get update
- $ sudo apt-get install python3-venv
- $ sudo apt-get install pip -y
- $ python3 -m venv .SchoolStudyRAGEnv

4. Launch VS Code and locate the same path (<PWD>\SchoolStudyRAG\)

5. Open terminal and activate the Python3 environment
- $ source .SchoolStudyRAGEnv/bin/activate

6. Launch the Qdrant docker image
- Please visit following link for Qdrant information,  https://qdrant.tech/documentation/quickstart/
- $ docker pull qdrant/qdrant
- $ docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v "$(pwd)/qdrant_storage:/qdrant/storage:z" qdrant/qdrant
	This will start the Qdrant server and same can be checked on http://localhost:6333/dashboard
- And for the next run just docker start qdrant

7. Launch the Ollama docker image
- $ docker pull ollama/ollama
- $ docker run -d --name ollama -p 11434:11434 -v /home/dipesh/SchoolStudyRAG/local_models:/root/.ollama ollama/ollama:latest
	This will start the ollama server and same can be checked on http://localhost:11434
- And for the next run just docker start ollama

9. Pull required model
- $ docker exec -it ollama ollama pull mxbai-embed-large
- $ docker exec -it ollama ollama pull gemma3:4b
- $ docker exec -it ollama ollama list
- $ docker exec -it ollama ollama pull gemma3:12b
- $ docker exec -it ollama ollama rm llama3.1:8b deepseek-r1:8b
- $ docker exec -it ollama ollama pull deepseek-r1:8b
llama3.1:8b
mistral:7b <Good one>
deepseek-r1:8b

7. Install required Python packages
- $ pip install -r requirements.txt