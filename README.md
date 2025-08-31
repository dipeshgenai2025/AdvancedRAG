# Advanced RAG Pipeline for PDFs

This project implements an **end-to-end Retrieval-Augmented Generation (RAG) pipeline** for PDFs. It extracts text and images, generates embeddings, stores them in a vector database, and allows semantic querying using a Large Language Model (LLM).

---

## Features

- **PDF Parsing:** Extracts text, images, and tables.
- **OCR for Images:** Optional OCR to retrieve text from images. [Optional]
- **Image Captioning:** Generates descriptive captions for images.
- **Text Preprocessing:** Splits text into sentences for embedding.
- **Embeddings Generation:** Converts text into vector representations.
- **Vector Database Storage:** Uses **Qdrant** for efficient semantic search.
- **Query Interface:** Retrieve top-K relevant sentences for a question.
- **LLM Integration:** Generates context-aware answers using **Ollama** or other LLMs.
- **Modular & Scalable:** Easily extendable for new ingestion formats or models.
- **GPU Support:** Embeddings and LLM inference can be GPU-accelerated.

---

## Architecture & Workflow

```mermaid
flowchart LR
    A[PDF Document] --> B[PDF Parser]
    B --> C[Extracted Text]
    B --> D[Extracted Images]
    D --> E[Image Captioning]
    E --> F[Combined Text + Captions]
    C --> F
    F --> G[Text Preprocessing & Sentence Splitting]
    G --> H[Embedding Generation]
    H --> I[Qdrant Vector Database]
    J[User Question] --> K[Embedder -> Query Qdrant]
    K --> L[Top-K Relevant Sentences]
    L --> M[LLM (Ollama) Generates Answer]
    M --> N[Final Answer]

---

# 🏗️ Directory Structure

```
.
├── Config
│   ├── __init__.py
│   └── config.py
├── Embeddings
│   ├── __init__.py
│   └── embedder.py
├── Ingestion
│   ├── __init__.py
│   ├── image_BlipCaptioner.py
│   ├── image_Captioner.py
│   ├── ocr.py
│   ├── pdf_parser.py
│   └── splitter.py
├── LICENSE
├── LLM
│   ├── __init__.py
│   └── ollama_client.py
├── Models
│   ├── EmbeddingModels
│   │   ├── Placeholder
│   │   └── mpnet-base-v2
│   ├── ImageCaptionModels
│   │   ├── Placeholder
│   │   └── blip
│   └── MainLLM
│       ├── Placeholder
├── QdrantDBStorage
│   ├── Placeholder
├── RAG_Pipeline
│   ├── RAG_Pipeline.py
│   ├── __init__.py
├── README.md
├── Retrieval
│   ├── __init__.py
│   └── retriever.py
├── TempData
│   └── Placeholder
├── Test
│   ├── Test.pdf
│   ├── __init__.py
│   ├── test_embedder.py
│   └── test_pdf_parser.py
├── Utils
│   ├── __init__.py
│   ├── logger.py
│   └── utils.py
├── Vectorstore
│   ├── __init__.py
│   └── qdrant_handler.py
├── main.py
└── requirements.txt
```

---

# Prerequsits
1. Create Linux distro OR VM on Linux or WSL on Windows to create a Linux runnable
2. Enable docker integration or installation
3. Git pull this repo
	- git clone https://github.com/dipeshgenai2025/AdvancedRAG

4. Open VS Code and sync with the working directory of Linux VM
5. Open terminal and create python environment
	- $ sudo apt-get update
	- $ sudo apt-get install python3-venv
	- $ sudo apt-get install python3-pip -y
  - $ sudo apt-get install tesseract-ocr -y
	- $ python3 -m venv .AdvancedRAG [Naming is up to the user]
	- $ source .AdvancedRAG/bin/activate [Naming is up to the user]

6. Install required Python packages
- $ pip3 install -r requirements.txt

7. Download the "Image to text" model and save offline
- git clone https://huggingface.co/Salesforce/blip-image-captioning-base into /AdvancedRAG/Models/ImageCaptionModels/blip

8. Download the "Caption generation" model and save offline
- git clone https://huggingface.co/sentence-transformers/all-mpnet-base-v2 into /AdvancedRAG/Models/EmbeddingModels/mpnet-base-v2

9. Launch the Qdrant docker image
- Please visit following link for Qdrant information,  https://qdrant.tech/documentation/quickstart/
- $ docker pull qdrant/qdrant
- $ docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v "$(pwd)/QdrantDBStorage:/qdrant/storage:z" qdrant/qdrant
- For the next run just "docker start qdrant"

10. Launch the Ollama docker image
- $ docker pull ollama/ollama
- $ docker run -d --name ollama --gpus all -p 11434:11434 -v $(pwd)/Models/MainLLM:/root/.ollama ollama/ollama:latest
	  This will start the ollama server and same can be checked on http://localhost:11434. For the next run just "docker start ollama"

11. Download any Ollama comapatible AI models
- $ docker exec -it ollama ollama pull gemma3:4b
- $ docker exec -it ollama ollama pull mistral:7b <Got better results>
- $ docker exec -it ollama ollama pull llama3.1:8b
- $ docker exec -it ollama ollama pull deepseek-r1:8b

12. Final execution
- $ python3 main.py