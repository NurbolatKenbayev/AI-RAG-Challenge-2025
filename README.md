# AI-RAG-Challenge-2025
Hackathon by Halyk Bank related to RAG


**Embedding model:**
https://huggingface.co/deepvk/USER2-base
It supports **Matryoshka Representation Learning (MRL)** — a technique that enables reducing embedding size with minimal loss in representation quality, which is useful for **quantization to reduce latency at the retriever step at the production**.

**Reranker:**
https://huggingface.co/jeffwan/mmarco-mMiniLMv2-L12-H384-v1


**Configuration**

Pre-launch steps:
- Download HuggingFace models (embedder, reranker) and provide paths in `.env` environment variables. 
- Run **Weaviate** instance in a Docker container (with mount for persistence)

When you launch **Docling** for the first time it automatically downloads the required models:
    * EasyOCR (in `~/.EasyOCR`) with 
        - craft_mlt_25k.pth (79MB) - Text detection model
        - latin_g2.pth (15MB) - Text recognition model for Latin scripts
