"# persian-cultural-rag-agent" 


a basic for making a RAG system step by step 


# Problem Statement

برای اینکه گردشگران بتوانند به اطلاعات دقیق تری از بناهای تاریخی ایران 
دستری داشته باشند و با ان اشنا شوند

```mermaid
graph TD
    subgraph Ingestion Pipeline [بخش آماده‌سازی داده]
        A[Raw Data] --> B[Cleaning & Schema Validation]
        B --> C[Chunking]
        C --> D[Embedding Model]
        D --> E[(Vector DB / Qdrant)]
    end

    subgraph Retrieval Pipeline [بخش جستجو]
        F[User Query] --> G[Embed Query]
        G --> H[Vector Search]
        E --> H
        H --> I[Top-K Context Chunks]
    end